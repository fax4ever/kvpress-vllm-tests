# KeyDiff KV Compression in vLLM — Final Implementation

This document records the actual implementation of KeyDiff KV cache
compression on the vLLM branch `fax-0.18` (repo: `/Users/fax/code/vllm`),
including every design decision, why it was made, which alternatives were
rejected, and what a future reader (human or agent) needs to know before
touching this code.

It complements the earlier research docs:

- `basic-ideas.md` — primitives and terminology
- `block-table-chain.md` — how block tables and `num_computed_tokens` flow
- `keydiff-vllm-integration.md` — Strategy A (full replacement) research
- `cache-filtering-approach.md` — Strategy B (append-only filtering) research

Those docs were corrected during Phase 1 to match the current branch and
remain the background reading. This document describes what was actually
built, which deviates from the research proposals in one important way
(see "Placement decision").

---

## 1. What was built

Two opt-in compression strategies, selected via configuration:

```python
# Strategy A — compact the prompt KV once, right after prefill:
llm = LLM(model="Qwen/Qwen3-8B",
          kv_compression_algorithm="full_replacement",
          kv_compression_ratio=0.5)

# Strategy B — online per-head keep/skip during decode:
llm = LLM(model="Qwen/Qwen3-8B",
          kv_compression_algorithm="filtering",
          kv_compression_ratio=0.5)
```

CLI equivalents: `--kv-compression-algorithm {full_replacement,filtering}`,
`--kv-compression-ratio 0.5` and `--kv-compression-interval N` (tokens
between retroactive compactions; default 512, the kvpress
`DecodingPress.compression_interval` default). When unset (the default),
every new code path is behind a `None`/zero check and behavior is
byte-identical to upstream.

**Phase coverage** — both algorithms cover all three phases, mapping onto
the configurations validated in the Phase-1 NIAH experiments:

| Phase | `full_replacement` | `filtering` |
|---|---|---|
| Prefill | retroactive compaction, unconditional at prefill end | same (≙ `PrefillDecodingPress` + retroactive KeyDiff) |
| Continuation (prefill chunks) | interval-based retroactive compaction at chunk boundaries (block-wise iterative, as in the KeyDiff paper) | same |
| Decode | interval-based retroactive compaction (≙ `CompressionRatioDecodingPress`) | per-head online keep/skip every step (≙ `FilteringPress`) |

Scoring is KeyDiff, implemented natively (no kvpress dependency): score of
a token = negative cosine similarity between its key and the per-head
anchor (mean of the L2-normalized keys). Validated to match
`kvpress.KeyDiffPress.score` exactly.

---

## 2. The one mechanism everything hangs on: `num_kv_discarded`

The research docs identified vLLM's core obstacle: a single per-request
sequence length serves four conflicting purposes (RoPE positions, slot
mapping, attention bounds, block allocation). Compression requires
splitting it into:

- **Logical position** — total tokens ever processed. Never decreases.
  Drives RoPE, token ids, stop conditions, the scheduler's
  `num_computed_tokens` cursor.
- **Physical cache position** — index into the request's cache slots.
  Drives slot mapping, attention `seq_lens` (`seqused_k` in flash-attn),
  and block allocation.

The entire implementation reduces this to **one integer per request**:

```
physical = logical - num_kv_discarded
```

Both strategies only ever *increase* `num_kv_discarded`:

- Strategy A sets it once at prefill end: `D = P - n_kept`.
- Strategy B increments it by 1 each time a decode step frees a cache
  column (no head of any layer extended past the previous max length —
  kvpress's "shrink when the trailing column is all padding").

This single-counter design works because both strategies keep the cache
**packed at the front**: valid entries always occupy cache positions
`[0, physical)`, with no holes. Full replacement scatters survivors into
the leading slots; filtering lets the next token overwrite the rejected
token's slot (the rejected token sat at the last position, so the packing
invariant is preserved).

### Two copies, one owner

- **Worker copy** (`CachedRequestState.num_kv_discarded` in
  `gpu_input_batch.py`) — **authoritative**. The worker makes the
  decisions (it is the only place with access to the keys) and uses this
  copy in `_prepare_inputs` to shift coordinates. Correctness of slot
  mapping and attention bounds depends only on this copy.
- **Scheduler copy** (`Request.num_kv_discarded` in `v1/request.py`) —
  advisory. Updated from `ModelRunnerOutput.kv_compression_discarded`
  (a new `dict[req_id, int]` field) in `Scheduler.update_from_output`.
  Used in exactly one place: `KVCacheManager.allocate_slots`, where
  occupancy is computed as `num_computed_tokens - num_kv_discarded`.

**Why this split is async-safe.** With async scheduling (now the default),
the scheduler may schedule step N+1 before consuming step N's output, so
its copy can be one step stale. A stale (smaller) value only makes the
scheduler *over*-allocate blocks — harmless. It can never be too large
(the counter is monotonic and only the worker increments it). Meanwhile
the worker executes steps strictly serially (`execute_model` calls are
sequential in the worker busy loop), so by the time `_prepare_inputs` runs
for step N+1, the post-forward hook of step N has already updated the
worker copy. This argument is why no synchronization or handshake is
needed anywhere.

### Preemption

On preemption the KV cache is freed and the request re-prefills from
scratch, so both copies reset to zero:

- Scheduler: `_preempt_request` sets `request.num_kv_discarded = 0`
  (next to the existing `num_computed_tokens = 0`).
- Worker: the resumed-from-preemption branch of `_update_states` resets
  `num_kv_discarded = 0` and `kv_compressed = False`. Re-prefill then
  re-compresses (Strategy A) or restarts filtering (Strategy B) —
  intended behavior, same result as the first time for A (deterministic
  count).

---

## 3. Placement decision: post-forward in the model runner

**This is the main deviation from the research docs**, which proposed
inserting compression inside the attention path, between the KV-cache
update op and the attention kernel. The implementation instead runs in
`GPUModelRunner.execute_model`, immediately after `self._model_forward`
returns. Three reasons, each individually sufficient:

1. **Prefill correctness (Strategy A).** kvpress compresses *after* a
   layer's attention has run. If compaction ran between cache-update and
   attention within the prefill forward pass, the prompt tokens would
   attend to an already-compacted cache and prefill logits would be
   wrong. Post-forward, the prefill step (and its sampled first token)
   used the full cache; only subsequent steps see the compacted cache.

2. **Shared metadata forces per-request decisions (Strategy B).** vLLM
   has one `seq_lens` entry and one block-table row per request, shared
   by *all* layers of a KV cache group. A layer-local keep/skip decision
   made inside layer 3's forward cannot differ from layer 20's — there is
   nowhere to record divergent lengths. So the decision must aggregate
   over all layers' scores, and all layers' keys only exist after the
   full forward completes.

3. **torch.compile / CUDA graphs.** The `unified_kv_cache_update` and
   `unified_attention_with_output` custom ops run inside the compiled /
   graph-captured region. Data-dependent eager logic (topk, `.item()`
   syncs, python loops over requests) does not belong there. Post-forward
   code is plain eager PyTorch, launched after graph replay on the same
   stream — no interaction with capture at all.

The spirit of the docs is preserved: compression still operates on the
paged cache via gather/score/scatter, and the metadata separation lands in
the scheduler + attention-metadata builder exactly as the "Metadata
Problem" section prescribed.

### Where the hook actually sits

```text
GPUModelRunner.execute_model()
  ├── _update_states()            # refreshes num_computed_tokens_cpu (pre-step values)
  ├── _prepare_inputs()           # ← coordinate shift applied here (reads worker copy)
  ├── _model_forward()            # all layers write K/V, attention runs
  ├── kv_compression_mgr.run_post_forward()   # ← NEW: decisions made here
  └── (returns; sample_tokens() then attaches the report to ModelRunnerOutput)
```

Timing detail that matters: `input_batch.num_computed_tokens_cpu[i]` holds
the *pre-step* value (tokens computed before this step), because positions
are built from it. The hook uses this plus
`scheduler_output.num_scheduled_tokens[req_id]` to classify each request:

- **A triggers** when `computed_before < prompt_len <= computed_before + scheduled`
  (the step that completes prefill, chunked or not) and `kv_compressed`
  is not yet set. It compacts *all* `computed_before + scheduled` cached
  tokens (== prompt length in the normal case).
- **B triggers** when `scheduled == 1 and computed_before >= prompt_len`
  (a pure decode step past the prompt).

---

## 4. The coordinate shift in `_prepare_inputs`

The precise integration point discovered during code reading
(`gpu_model_runner.py`, `_prepare_inputs`): positions, slot mapping and
seq_lens all derive from `num_computed_tokens_cpu`:

```python
positions_np = num_computed_tokens_cpu[req_indices] + arange   # logical
block_table.compute_slot_mapping(req_indices, positions_np)    # ← physical needed
seq_lens = num_computed_tokens_cpu + num_scheduled_tokens      # ← physical needed
```

The patch (guarded by `kv_compression_mgr is not None`, and skipped
entirely when all counters are zero):

1. Build `kv_discarded_np` from the worker-authoritative per-request
   counters (a small python loop over `input_batch.req_ids` — O(num_reqs),
   only when compression is enabled; deliberately avoids adding another
   persistent array to `InputBatch` that would need maintenance in
   `add_request`/`swap_states`/`condense`).
2. `cache_positions = positions_np - kv_discarded_np[req_indices]` →
   passed to `compute_slot_mapping`. **`positions_np` itself stays
   logical** — it also feeds RoPE and token-id gathering.
3. `seq_lens -= kv_discarded_np` — but only **after** computing
   `discard_request_mask` (which decides which requests get sampled by
   comparing seq_lens to total token counts and must use *logical*
   lengths; using physical lengths there would permanently mark
   compressed decode requests as "partial prefill" and they would never
   be sampled). This required reordering the mask computation before the
   subtraction; when compression is off the reorder is a no-op.

Consumers checked and left alone deliberately:

- `max_seq_len` (attn metadata) now derives from adjusted seq_lens —
  correct, it becomes flash-attn's `max_seqlen_k`.
- `CommonAttentionMetadata._num_computed_tokens_cpu` stays logical; the
  only consumer on our path is batch reordering (decode-vs-prefill
  classification), where logical is semantically right.
- The `seq_lens` GPU buffer is a persistent CpuGpuBuffer — mutating its
  *contents* is CUDA-graph-safe (it is a graph input buffer); replacing
  the tensor would not have been.

---

## 5. Strategy A internals (`compact_request_kv` + interval triggers)

Strategy A is retroactive compaction covering **all phases** — the vLLM
port of kvpress `CompressionRatioDecodingPress` (contributed upstream as
PR #231), extended to chunked prefill. The trigger rule, evaluated
post-forward for every scheduled request:

```text
compact when   logical_tokens_now - last_compaction_total >= interval
               (any phase: prefill chunk, continuation chunk, or decode)
or when        prefill completes this step (unconditional)

target         n_kept = max(1, int(logical_total * (1 - ratio)))
               # CompressionRatioDecodingPress._resolve_target_size:
               # a fraction of ALL tokens seen so far, including
               # previously discarded ones
```

`interval` is `--kv-compression-interval` (default 512 = kvpress
`DecodingPress.compression_interval` default). Compacting at prefill
*chunk* boundaries has no kvpress reference (HF has no chunked prefill),
but it is exactly block-wise iterative compression — the scheme of the
original KeyDiff paper (kvpress `BlockPress`). Per-request state:
`CachedRequestState.kv_last_compaction_total` (logical count at the last
compaction; reset on preemption).

Note that with the default interval of 512, short decode runs will show
**no** decode-phase compaction — lower the interval (e.g. 16) in
experiments where you want to observe it.

The compaction itself, per request, per layer:

```text
src_slots = slots for cache positions [0, T)          # T = currently cached tokens
keys      = key_cache[blk, off]                       # [T, H, D] advanced indexing
scores    = keydiff(keys)                             # [H, T]
idx       = scores.topk(n_kept).indices.sort()        # per head, temporal order kept
scatter survivors (gather per head) into slots [0, n_kept)
```

Re-compaction of an already-compacted cache is valid: the cache holds
composite per-head entries (column i = different token per head), and
scoring/selection are fully per-head, so each head's row is just a valid
sequence of that head's kept K/V.

Notes and choices:

- **Deterministic count.** With a fixed ratio, every head and every layer
  keeps the same *count* (different *sets*) — this is the property that
  makes a single shared `seq_lens` valid, and it also means Strategy A is
  TP-safe (each rank compacts its own heads to the same length) and needs
  no worker→scheduler synchronization beyond the ordinary report.
- **Indexing, not `reshape_and_cache_flash`.** Gather/scatter use plain
  advanced indexing `cache[slot // bs, slot % bs]`. This works regardless
  of the physical stride order (NHD/HND layouts permute strides, not the
  logical shape), works on CPU (unit-testable without a GPU), and avoids
  the custom op's quantization-scale arguments (fp8 caches are rejected
  at config validation anyway).
- **Sorted kept indices** preserve temporal order in the cache. Attention
  is permutation-invariant over the KV set so this is not required for
  correctness, but it keeps the cache state comprehensible and would
  matter if windowed variants were ever added.
- **Stale tail.** Slots `[n_kept, T)` keep stale data. They are never read
  (seq_lens bounds attention) and are progressively overwritten by new
  decode tokens at positions `n_kept, n_kept+1, ...`.
- **Mid-prefill compaction is safe.** No sampling happens mid-prefill;
  the next chunk's queries attend over the compacted prefix — which is
  the *intended* semantics of block-wise iterative compression, not an
  artifact. (It does mean prompt logprobs differ from the uncompressed
  baseline, as compression inherently must.)

## 6. Strategy B internals (`filtering_step`)

**True per-head FilteringPress** — a faithful port of
`kvpress.FilteringPress` + `PaddedTensor` onto the paged cache. No voting,
no aggregation: every (layer, head) decides independently and keeps its
own token set. (An earlier revision of this implementation collapsed the
per-head votes into a single per-token majority decision —
`UniformFilteringPress`'s structure — because per-head ragged lengths
looked unrepresentable in vLLM. The insight that unlocked the faithful
version is below.)

**Prefill coverage.** Strategy B is the vLLM equivalent of the validated
NIAH configuration `PrefillDecodingPress(prefilling_press=KeyDiffPress,
decoding_press=FilteringPress)`: during the prefill phase it runs the
same retroactive compaction as Strategy A (interval-based at chunk
boundaries, unconditional when prefill completes), then hands off to
per-head online filtering for decode. The handoff is seamless — after the
prefill-end compaction the cache is fully packed at the compacted length,
so `kv_filter_lengths` initializes to that length for every (layer, head)
at the first filtered decode step (`kv_compressed` marks the handoff).

### The key insight: `PaddedTensor`'s packing gives a shared bound

What makes kvpress's FilteringPress genuinely per-head is not the votes —
it is the per-head *packing*. Each head's accepted tokens are packed at
the front of the shared buffer (`accept_last` copies the new token's slice
into the head's first padding column), so buffer column `i` holds a
*different token* for different heads, and the buffer only shrinks when a
trailing column is padding for **all** heads. Consequence: the physical
buffer length is always `max_h L_h` — **one scalar**. And one scalar is
exactly what vLLM's shared `seq_lens` can represent:

```
shared physical length C  =  max over (layer, head) of L[layer][head]
num_kv_discarded          =  logical_total - C
```

Heads (and layers) whose `L < C` attend over a few stale trailing columns
— exactly kvpress's `fill_padding=False` variant, which the Phase-1 NIAH
experiments showed is quality-equivalent to zero-filling. Layers share
one seq_len in vLLM (kvpress buffers are per-layer), so the max also runs
across layers; a layer behind the global max just has a few more stale
columns. Per-layer dynamics are otherwise fully independent and
column-for-column identical to kvpress (the new token enters at the
shared last column instead of the layer's own last column, but the *set*
of scored values and the copy destination `L_h` are the same).

### Per decode step (post-forward)

```text
The new token was already written at shared cache column C-1
(C = num_cached incl. new) and this step's attention already saw it —
matching FilteringPress, where attention includes the token and
filtering happens afterwards.

n_kept = clamp(round(logical_total * (1 - ratio)), 1, C)
           # logical_total counts skipped tokens!
per layer, vectorized over heads:
    valid_h  = columns [0, L_h) ∪ {C-1}          # head's packed prefix + new token
    scores   = masked KeyDiff (per-head anchor over valid only; -inf elsewhere)
    thresh_h = n_kept-th highest score            # -inf if head has < n_kept valid
    accept_h = score_new >= thresh_h
    if accept_h: copy new token's per-head K/V slice to column L_h
                 (PaddedTensor.accept_last); L_h += 1
new C = max(L); delta = old C + 1 - new C ∈ {0, 1}
if delta == 1: num_kv_discarded += 1              # column freed — kvpress shrink()
```

Per-request state: `CachedRequestState.kv_filter_lengths`, a
`[num_layers, num_kv_heads]` int tensor on the cache device, lazily
initialized to the fully-packed pre-step length at the first filtered
decode step, reset to `None` on preemption/resume.

- **Why `logical_total` (not cached count) in `n_kept`:** cached tokens
  are self-selected high scorers; calibrating the threshold to them alone
  is far too strict (0.1% keep rate in the Monte Carlo study,
  `vllm-experiments/05a_filtering_bias_analysis.md`). Using the logical
  total — as if skipped tokens were still competing — plus `round()`
  gives ~50.3% realized keep rate at target 50%. This is the
  self-regulating feedback loop that makes the realized ratio converge to
  the target in expectation.
- **Column freed ⇔ all heads of all layers rejected past it.** If even
  one head accepts while at the current max, the shared length grows
  (delta 0). If accepting heads all sit below the max (gaps from earlier
  rejections), they copy the token into their own columns and the
  trailing column is still freed (delta 1) — the token survives *only*
  inside the accepting heads' packed prefixes. Identical to
  `accept_last` + `shrink`.
- **Sync-free copies.** Rejected heads perform a harmless self-copy of
  the last column (`torch.where` on the destination) so no
  `.nonzero()`/`.any()` GPU→CPU sync is needed per layer; the only sync
  is the final `lengths.max().item()` per request per step.
- **Chunked-prefill continuation is compacted, not filtered.** Prompt
  chunks are cached in full as they arrive and compressed retroactively
  at interval boundaries and at prefill end (see "Prefill coverage"
  above); per-token online filtering of chunk tokens is not implemented
  (kvpress never validated it, and retroactive compaction of the chunk is
  the better-informed operation anyway — it sees the whole prefix).
- **Cost:** every decode step gathers and scores *all* cached keys for
  *every* layer (per-head masking adds a normalize+mask pass), plus one
  GPU→CPU sync per filtered request per step. Faithful ("scores the full
  cache") but expensive — the known optimization targets are batching
  across requests/layers and incremental anchor maintenance.
- **Memory overhead of the max-bound:** the shared length is the max over
  all (layer, head) lengths, so memory sits slightly above the per-head
  target. Empirically small: in the 1200-step CPU run, per-head keep
  ratio 0.538 vs shared/memory ratio 0.553 (target 0.5) — head lengths
  track each other closely because they share the same feedback target.

### Behavioral equivalence with FilteringPress + PaddedTensor: what is guaranteed, and how

No kvpress code is imported into vLLM — kvpress is not a dependency, and
`PaddedTensor` could not be reused anyway (it wraps a dense, contiguous
`[batch, heads, seq, dim]` tensor it can clone, slice and physically
shrink; the paged cache is scattered across fixed-size blocks, per layer,
and can do none of that). The port instead decomposes PaddedTensor's
state and operations onto vLLM's structures:

| kvpress `FilteringPress` + `PaddedTensor` | vLLM equivalent |
|---|---|
| `.data` (dense per-layer buffer) | the paged cache itself, addressed via block-table slots |
| `.lengths` `[batch, heads]`, one per layer (`fp._lengths[layer]`) | `CachedRequestState.kv_filter_lengths` `[num_layers, num_kv_heads]` |
| `valid_mask(include_last=True)` | built inline in `filtering_step` (`col < L_h`, last column forced valid) |
| per-head `base_press.score` on valid keys | `masked_keydiff_scores` (per-head anchor over valid keys only, `-inf` elsewhere) |
| `n_kept = round(total_seen·(1−r))`, clamp | same formula, same clamp semantics (see note 2b below) |
| `accept_last(accepted)` | per-head K/V slice copy into column `L_h` + `lengths += accepts` |
| `shrink()` | shared length = `lengths.max()`; a freed trailing column → `num_kv_discarded += 1` |
| `fill_padding()` | intentionally omitted — the stale variant |

**The exact claim.** Given the same per-step key/value stream, the
cache-*state* trajectory is identical to FilteringPress with
`fill_padding=False`: after every decode step, (i) every (layer, head)
valid length equals kvpress's, (ii) every head's packed prefix — the K/V
values at that head's columns `[0, L_h)` — equals kvpress's buffer
content column-for-column, and (iii) the shared physical length equals
the max kvpress buffer length across layers. In other words, the state
machine is the same machine; only its storage substrate differs.

**How the guarantee is established — three pillars:**

1. **Unit-level scoring equivalence.** `keydiff_scores` is verified exact
   against `KeyDiffPress.score`, and `masked_keydiff_scores` is verified
   exact against running the scorer per head on only that head's valid
   keys — which is literally what `FilteringPress.compress` does (it
   loops `head_keys = kt.data[b, h, valid_pos]` and scores those). Same
   keys in, same anchor, same scores out.

2. **An operation-by-operation mapping argument** for the two places the
   implementations *look* different but provably are not:

   a. *The new token's column.* kvpress appends the token at each layer
      buffer's own last column (`max_h L_h` of that layer); vLLM writes
      it once at the shared last column (`max` across all layers). For a
      layer behind the global max these column indices differ — but the
      *set of scored values* per head (packed prefix + new token) is
      identical, the copy destination (`L_h`, the head's first free
      column) is identical, and the increment rule is identical. Where
      kvpress skips the copy for a head already at its layer's last
      column, vLLM performs a harmless self-copy — the resulting cache
      contents are the same either way.

   b. *The clamp bound on `n_kept`.* kvpress clamps to its layer buffer
      width, vLLM to the (possibly larger) shared width. Whenever the two
      clamps produce different values, both land in the `-inf` padding
      region of the sorted scores — a head with `L_h + 1` valid tokens
      gets threshold `-inf` (always accept) under either bound, so the
      accept decision is unchanged in every branch.

3. **A stateful differential test as the standing regression gate**
   (`test_filtering_step_multistep_vs_kvpress`): 60 decode steps × 2
   layers drive the *real* `FilteringPress` + `PaddedTensor` (from the
   kvpress fork with PR #257) and `filtering_step` on the same random
   K/V stream, asserting claims (i)–(iii) with exact tensor equality
   after **every** step — through keeps, skips, head divergence and
   gap-filling copies (30/60 columns freed in the run; per-head lengths
   diverged to e.g. `[48, 53, 50, 54]`). Any future edit to
   `filtering_step` that drifts from PaddedTensor semantics fails this
   test on the first divergent step.

**What is *not* claimed identical** — and why it is bounded: end-to-end
generated tokens are not bit-identical to an HF+kvpress run, for two
reasons that have nothing to do with the filtering state machine.
First, attention *visibility* of stale columns differs slightly: kvpress
bounds each layer's attention by that layer's own buffer length, while
vLLM bounds all layers by the shared cross-layer max, so a lagging layer
attends over a few extra stale columns. This is the same regime as
kvpress's `fill_padding=False` (stale) variant — Phase-1 NIAH showed
stale vs zero-filled padding is quality-neutral — extended across layers.
Second, kernel numerics differ (paged flash-attention vs HF eager).
Neither affects which tokens are kept given the same keys; they affect
only the attention outputs computed *over* the kept tokens.

---

## 7. File-by-file change map

New:

| File | Contents |
|------|----------|
| `vllm/v1/worker/kv_compression.py` | `keydiff_scores`, `masked_keydiff_scores`, `gather_slots`/`scatter_slots`, `compact_request_kv`, `filtering_step`, `KVCompressionManager` (bind + post-forward driver) |
| `tests/v1/worker/test_kv_compression_standalone.py` | CPU math tests; cross-checks vs kvpress when importable; runnable without a vLLM install (stubs vllm-internal imports) |
| `tests/v1/core/test_kv_compression_scheduler.py` | Scheduler accounting: block savings for A and B, preemption reset |

Modified:

| File | Change |
|------|--------|
| `vllm/config/cache.py` | `kv_compression_algorithm` / `kv_compression_ratio` / `kv_compression_interval` fields + validator (ratio ∈ (0,1) iff algorithm set; interval ≥ 1; no fp8 cache); all excluded from the compile-graph hash (eager post-forward work, no graph impact) |
| `vllm/config/vllm.py` | Cross-config validation (see §8); force-disables prefix caching |
| `vllm/engine/arg_utils.py` | EngineArgs fields, CLI args, prefix-caching force-off with warning, pass-through to CacheConfig |
| `vllm/v1/request.py` | `Request.num_kv_discarded` (scheduler copy) |
| `vllm/v1/core/sched/scheduler.py` | Reset on preemption; apply reported deltas in `update_from_output` |
| `vllm/v1/core/kv_cache_manager.py` | `allocate_slots`: occupancy = `num_computed_tokens - num_kv_discarded` (single change point; everything downstream — `num_tokens_need_slot`, `get_num_blocks_to_allocate`, `allocate_new_blocks` — flows from it) |
| `vllm/v1/outputs.py` | `ModelRunnerOutput.kv_compression_discarded: dict[str, int] \| None` |
| `vllm/v1/worker/gpu_input_batch.py` | `CachedRequestState.num_kv_discarded` / `.kv_compressed` / `.kv_filter_lengths` / `.kv_last_compaction_total` (worker copies; recreated at defaults when a request is re-added) |
| `vllm/v1/worker/gpu_model_runner.py` | Manager init; KV cache binding in `initialize_kv_cache_tensors` (after `bind_kv_cache`); `_prepare_inputs` coordinate shift; resume-from-preemption reset; post-forward hook; report attach + clear in `sample_tokens` |

The KV cache binding validates: exactly one KV cache group with
`FullAttentionSpec`, and every layer tensor shaped
`[2, num_blocks, block_size, H, D]` (the FLASH_ATTN layout — this shape
check is the real backend gate; FlashInfer's `[nb, 2, bs, H, D]` and MLA
layouts fail with an instructive error). Layers listed in
`shared_kv_cache_layers` (KV sharing) are skipped and tensors deduped by
`id()` so a shared cache is never compacted twice (compaction is not
idempotent).

---

## 8. Supported / rejected configurations and why

Enforced in `VllmConfig.__post_init__` (fail-fast at engine construction):

| Constraint | Reason |
|------------|--------|
| No speculative decoding | Multi-token decode steps break B's `scheduled == 1` model; rejected-draft rollback interacts with the discard counter in untested ways |
| `pipeline_parallel_size == 1` | The post-forward hook and output reporting assume a single worker produces `ModelRunnerOutput`; non-last PP ranks return early |
| No context parallelism (DCP/PCP) | Slot mapping uses interleaved virtual blocks; the coordinate shift was not designed for it |
| `tensor_parallel_size == 1` **for filtering only** | Filtering decisions are data-dependent per rank (each rank holds different heads); divergent decisions would desync per-rank cache coordinates. A is allowed with TP: its kept *count* is deterministic, so all ranks stay consistent |
| No pooling / encoder-decoder models | Pooling uses seq_lens for cursor building; encoder-decoder has cross-attention caches — both out of scope |
| No fp8 KV cache | Gather/scatter via plain indexing bypasses quantization scales |
| Prefix caching force-disabled | Compaction destroys slot↔token coherence in shared blocks (the CoW problem from the research doc); filtering changes what a "cached prefix" contains. Disabling removes block sharing entirely (`ref_cnt` always 1), which also guarantees cascade attention never triggers (no common prefix blocks) |
| Single full-attention KV group | No sliding window (its `remove_skipped_blocks` logic assumes position-aligned slots), no hybrid/Mamba models |

Async scheduling is **allowed** (see the staleness argument in §2).

---

## 9. What is deliberately NOT done (trade-offs)

1. **No tail-block freeing after compaction (Strategy A).** The worker's
   block-table rows are append-only (`append_row` extends; only
   resume-from-preemption replaces a row). Scheduler-side truncation
   would desync the two copies — the runner would keep stale block ids
   and compute wrong slots when new blocks arrive. Rather than invent a
   truncation protocol, v1 keeps the blocks: allocation is
   occupancy-based, so the request allocates **no new blocks** until its
   physical occupancy regrows past the original prompt length. Savings
   are real but deferred; immediate freeing is the highest-value follow-up
   and would need a scheduler→worker "row replaced" path (the
   resumed-request machinery is the natural template).
2. **Zero-filling stale columns (filtering).** Heads/layers behind the
   shared max length attend over stale trailing columns (kvpress
   `fill_padding=False`). Phase-1 showed zero-fill vs stale is
   quality-neutral, and zero-filling per head per step would cost extra
   writes, so it is skipped.
3. **Per-token online filtering of continuation chunks.** Chunks are
   covered by retroactive compaction instead (see §6) — kvpress never
   validated chunk-level online filtering, and retroactive compaction of
   the chunk has the better (global-prefix) view anyway.
4. **Memory-pressure-triggered compaction.** The interval trigger is
   token-count-based; reacting to block-pool pressure would need a
   scheduler→worker signal. The mechanism (report another delta)
   supports it; the trigger policy is left for later.
5. **Batched compression math.** Notebook 06 validated fully batched
   scoring/selection; the vLLM hook loops per request per layer for
   clarity. Fine while few requests cross the prefill boundary per step.

---

## 10. Validation performed (and not performed)

Local machine is a MacBook Pro — **no CUDA, no flash-attn** — so
end-to-end GPU validation is done separately on cloud NVIDIA GPUs.
What was validated locally:

- **Math vs kvpress reference** (the strongest signal):
  `tests/v1/worker/test_kv_compression_standalone.py`, run with the
  kvpress fork's venv (`/Users/fax/code/kvpress/.venv/bin/python`, which
  has torch 2.12 + the `FilteringPress` code from PR #257):
  - `keydiff_scores` ≡ `KeyDiffPress.score` (exact); `masked_keydiff_scores`
    ≡ per-head scoring on only the valid keys (exact).
  - Compaction ≡ per-head top-k reference on a 3-layer paged cache with
    non-trivial block tables (exact).
  - **Filtering: stateful multi-step equivalence with the real
    `FilteringPress`** — 60 decode steps × 2 layers, comparing after every
    step the per-(layer, head) lengths, the per-head packed K/V contents
    column-for-column, and the shared physical length vs the max kvpress
    buffer length. Exact match throughout (30/60 columns freed, per-head
    lengths genuinely diverged, e.g. `[48, 53, 50, 54]`).
  - Long-run ratio convergence: 1200 decode steps at target ratio 0.5 →
    per-head keep ratio 0.538 (matches the ~50.3% Monte Carlo figure in
    05a_filtering_bias_analysis.md), shared/memory ratio 0.553.
  - Iterative compaction across phases (`test_iterative_compaction`):
    prefill chunk → continuation chunk → three decode-phase compactions,
    each verified against an independently maintained per-head reference
    list (composite-cache re-compaction correctness).
  - Manager trigger logic (`test_manager_phase_coverage`): for both
    algorithms, chunked prefill fires interval compactions, prefill end
    fires the unconditional compaction, and decode fires interval
    compaction (A) / per-step per-head filtering (B), with exact
    discard-delta accounting at every step.
  - Gather/scatter round-trips bit-exact.
  The test file stubs `vllm.logger` / `vllm.v1.kv_cache_interface`, so it
  runs with or without a vLLM install:
  `python tests/v1/worker/test_kv_compression_standalone.py`.
- **Scheduler accounting**
  (`tests/v1/core/test_kv_compression_scheduler.py`, 3 passed):
  full-replacement block savings over 110 decode steps (10 blocks vs 14
  uncompressed), filtering savings (3 vs 4), preemption reset. Note the
  off-by-one that bit me writing these: `allocate_slots` sizes blocks for
  occupancy *at scheduling time* + new tokens, not post-step occupancy.
- **Regressions**: `tests/v1/core/test_scheduler.py` 93 passed,
  `test_prefix_caching.py` 49 passed, `test_async_scheduler.py` +
  `test_output.py` 10 passed. Config plumbing (CLI parse, LLM kwargs,
  validation errors, prefix-caching force-off) verified by direct
  invocation. `ruff check` / `ruff format` clean. All touched modules
  import cleanly (deps installed into the repo `.venv` from
  `requirements/common.txt` + CPU torch; note `vllm._C` is absent on mac,
  which is fine for these tests).

**Not validated (your GPU notebook's job):**

1. Baseline equivalence: with the option unset, outputs must be identical
   to upstream (they should be — every change is behind a guard — but
   this is the claim to verify first).
2. Coherent generations at ratio 0.25–0.5 for both algorithms on a real
   model (Qwen3-8B on A100 is the reference setup from the notebooks).
3. Memory effect: watch block usage (e.g. `num_gpu_blocks` metrics or
   preemption behavior under memory pressure) to see the deferred-savings
   profile of A vs the immediate profile of B.
4. NIAH benchmark vs the kvpress numbers (the end-to-end quality loop
   from `cache-filtering-approach.md` Step 4).

Practical GPU-notebook tips:

- If the platform picks a different backend, force it:
  `--attention-backend FLASH_ATTN` (or env `VLLM_ATTENTION_BACKEND`).
  A wrong backend fails fast at KV-cache bind with a shape-layout error.
- `enforce_eager=True` is a useful first run to take CUDA graphs out of
  the equation; then re-run with graphs on — behavior should be identical
  (the compression hook is outside the captured region either way).
- To observe filtering decisions, log
  `ModelRunnerOutput.kv_compression_discarded` or watch
  `Request.num_kv_discarded` grow; realized ratio ≈
  `num_kv_discarded / num_output_tokens` after long generations.

---

## 11. Non-obvious internals a future agent must know

Hard-won facts about this vLLM branch that the implementation depends on
(re-verify these if rebasing far):

1. **The FA cache-update is a separate custom op.**
   `FlashAttentionBackend.forward_includes_kv_cache_update = False`;
   `Attention.forward` calls `torch.ops.vllm.unified_kv_cache_update`
   (→ `impl.do_kv_cache_update` → `reshape_and_cache_flash`) *before*
   `unified_attention_with_output`. This is why, on a decode step, the
   new token is already in the cache when attention runs — filtering's
   semantics rely on this ordering.
2. **`input_batch.num_computed_tokens_cpu` holds pre-step values** (the
   scheduler sends the cursor as of before the step; it increments its
   own copy at schedule time, `scheduler.py:964`). All hook eligibility
   logic depends on this.
3. **Worker execution is strictly serial** even under async scheduling
   and batch queues — one busy-loop thread, sequential `execute_model`
   calls. The whole no-synchronization design rests on this.
4. **`discard_request_mask`** (`_prepare_inputs`) compares seq_lens
   against total token counts to skip sampling for partial prefills. It
   must see logical lengths; this is why the mask computation was moved
   above the seq_lens adjustment.
5. **`AsyncGPUModelRunnerOutput.get_output` mutates and returns the same
   `ModelRunnerOutput`** — added fields pass through untouched.
6. **`kv_caches` dict contains aliases for KV-sharing layers**
   (`shared_kv_cache_layers`); `bind_kv_caches` must dedupe or compaction
   would run twice on the same tensor.
7. **`cache_blocks` is a no-op when `enable_caching=False`** — this is
   what makes force-disabling prefix caching sufficient; no other
   caching-path guards were needed (including the `AsyncScheduler`
   subclass, which inherits both patched methods).
8. **`_dummy_run` / profiling never call `_prepare_inputs`** with real
   requests, so the compression code paths are inert during memory
   profiling and CUDA-graph capture.
9. **Line anchors** (this branch): coordinate shift in `_prepare_inputs`
   around `gpu_model_runner.py:1793–1850`; post-forward hook ~`:3840`;
   report attach ~`:4130`; cache bind ~`:6497`; allocation change
   `kv_cache_manager.py:314–323`; scheduler apply
   `scheduler.py:1306–1318`.

---

## 12. Summary of assumptions made (flagged, not silently chosen)

1. **Post-forward placement** instead of the docs' in-layer insertion
   (rationale in §3) — the docs' layer identification was honored where
   it holds (paged-cache gather/scatter, scheduler/allocator metadata),
   corrected where the constraint analysis showed it cannot work.
2. **Stale trailing columns for lagging heads/layers** in filtering
   (kvpress `fill_padding=False` semantics): the per-head ragged lengths
   are real (full FilteringPress fidelity), but heads/layers below the
   shared max length attend over a few stale columns. Phase-1 showed
   stale ≈ zero-fill in quality; the shared max also runs across layers
   (kvpress buffers are per-layer), which is a vLLM-specific extension of
   the same idea.
3. **Per-token online filtering runs only on single-token decode steps**;
   prefill/continuation chunks are covered by retroactive compaction
   instead (the better-informed operation; kvpress never validated
   chunk-level online filtering).
4. **No tail-block freeing** after compaction; savings are
   allocation-deferral, not immediate release.
5. **Compaction at prefill *chunk* boundaries has no kvpress reference**
   (HF has no chunked prefill) — it is justified as block-wise iterative
   compression, the original KeyDiff paper's scheme (kvpress
   `BlockPress`), but is not covered by the Phase-1 NIAH validation.
6. **`kv_compression_interval` default is 512** (kvpress
   `DecodingPress.compression_interval` default) — faithful, but it means
   short decode runs show no decode-phase compaction for
   `full_replacement` unless lowered.
7. **`n_kept` formulas**: retroactive compaction uses
   `max(1, int(logical·(1−r)))` (kvpress
   `CompressionRatioDecodingPress._resolve_target_size`); filtering uses
   `clamp(round(logical·(1−r)), 1, C)` (kvpress `FilteringPress`). They
   intentionally differ because the upstream references differ.

---

## 13. Why `flash_attn.py` is untouched (yes, this runs on CUDA FlashAttention)

A natural first reaction to the diff is: "no changes to
`vllm/v1/attention/backends/flash_attn.py` — does this actually work on
the CUDA FlashAttention backend?" Yes — and the absence of backend
changes is deliberate. It is the payoff of the observation in
`keydiff-vllm-integration.md` ("NVIDIA / Flash Attention 2"): the
FlashAttention kernel is a compiled CUDA binary we cannot modify, but we
do not need to, because everything it does is parameterized by metadata
we control upstream.

`FlashAttentionImpl.forward` consumes three inputs that fully determine
what the kernel touches:

- **`seqused_k` (= `attn_metadata.seq_lens`)** — how many KV entries per
  sequence the kernel iterates over. We shrink this in `_prepare_inputs`
  (`seq_lens -= kv_discarded`), so after compaction/filtering the kernel
  simply reads fewer entries. It has no idea compression happened.
- **`block_table`** — which physical blocks hold the sequence. Untouched:
  both strategies keep survivors packed in the leading slots of the
  *same* blocks.
- **`slot_mapping`** (consumed by `reshape_and_cache_flash` inside
  `do_kv_cache_update`) — where new tokens get written. We shift the
  positions that feed `compute_slot_mapping`, so the existing write op
  lands tokens at the compacted positions without knowing why.

So the backend sees a perfectly ordinary, slightly-shorter sequence every
step. The compression itself (gather → KeyDiff score → select → scatter)
never goes through the kernel at all — it is plain tensor indexing on the
paged cache tensors, done post-forward in the model runner (§3).

The two actual couplings to FlashAttention are:

1. **Cache layout.** The code assumes
   `[2, num_blocks, block_size, num_kv_heads, head_size]`, which is
   FLASH_ATTN's `get_kv_cache_shape`. Enforced at KV-cache bind time in
   `KVCompressionManager.bind_kv_caches`: any other backend (FlashInfer's
   `[nb, 2, bs, H, D]`, MLA layouts) fails at startup with an error
   telling you to use `--attention-backend FLASH_ATTN`. On A100/H100 with
   a model like Qwen3-8B, FLASH_ATTN is the default selection anyway.
2. **Write-before-attention ordering.** On this branch FLASH_ATTN has
   `forward_includes_kv_cache_update = False`, so the new token's K/V is
   in the cache before the attention op runs. Strategy B's semantics
   (token attends to itself, gets filtered afterwards) rely on that
   ordering (§11.1).

Note that even per-head ragged lengths (filtering, §6) did *not* require
touching the kernel: the PaddedTensor packing invariant means the shared
bound is simply `max over (layer, head) lengths`, and heads below the max
attend over a few stale columns (the quality-neutral
`fill_padding=False` variant). The only thing the kernel's
single-scalar-bound-per-sequence interface genuinely cannot express is
per-head *masking* of those stale columns — if zero-tolerance for stale
attention were ever required, that would be the first (and so far only)
reason to modify the backend.
