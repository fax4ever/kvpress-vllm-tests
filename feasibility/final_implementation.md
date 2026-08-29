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

# Strategy B — online per-token keep/skip during decode:
llm = LLM(model="Qwen/Qwen3-8B",
          kv_compression_algorithm="filtering",
          kv_compression_ratio=0.5)
```

CLI equivalents: `--kv-compression-algorithm {full_replacement,filtering}`
and `--kv-compression-ratio 0.5`. When unset (the default), every new code
path is behind a `None`/zero check and behavior is byte-identical to
upstream.

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
- Strategy B increments it by 1 each time a decode token is rejected.

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

## 5. Strategy A internals (`compact_request_kv`)

Per request, per layer (loop over the unique layer cache tensors):

```text
src_slots = slots for cache positions [0, T)          # T = tokens cached at prefill end
keys      = key_cache[blk, off]                       # [T, H, D] advanced indexing
scores    = keydiff(keys)                             # [H, T]
n_kept    = max(1, int(T * (1 - ratio)))              # kvpress ScorerPress formula
idx       = scores.topk(n_kept).indices.sort()        # per head, temporal order kept
scatter survivors (gather per head) into slots [0, n_kept)
```

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
- **Compress-once.** `kv_compressed` flag prevents re-compaction. Decode
  tokens are never retroactively compressed (mirrors kvpress prefill
  compression). Periodic re-compaction would be a natural extension: the
  mechanism (report a new discard delta) already supports it.

## 6. Strategy B internals (`filtering_keep_decision`)

Faithful port of `kvpress.FilteringPress` (verified vote-for-vote against
the real class from the local kvpress fork, branch with PR #257):

```text
The new token was already written at cache position C-1 (C = cached incl. new)
and this step's attention already saw it — matching FilteringPress, where
attention includes the token and filtering happens afterwards.

n_kept    = round(logical_total * (1 - ratio))   # logical_total counts skipped tokens!
n_kept    = clamp(n_kept, 1, C)
if n_kept >= C: keep (threshold would be the min score) — early exit, no gather
per layer:  scores = keydiff(gathered keys [C, H, D])
            threshold_h = n_kept-th highest score
            vote_h      = score_new >= threshold_h
decision  = majority vote over all (layer, head) pairs
```

- **Why `logical_total` (not cached count) in `n_kept`:** this is the
  self-regulating feedback loop from FilteringPress. If the cache grows
  above `(1-ratio) * logical`, the threshold rises and more tokens get
  rejected; if it falls below, `n_kept >= C` and everything is accepted.
  The realized ratio converges to the target in expectation.
- **If rejected:** nothing is un-written. `num_kv_discarded += 1` means
  the *next* token's cache position equals the rejected token's slot, so
  it gets overwritten one step later, and next step's `seq_lens` excludes
  it. The rejected token was visible to exactly one attention step (its
  own) — identical to FilteringPress semantics.
- **Majority vote is a deviation** from the validated per-head algorithm.
  Per-head ragged lengths are unrepresentable in vLLM (one seq_len per
  request). This is the single biggest quality risk: the Phase-1 NIAH
  results show per-head decisions matter at 75% compression
  (`UniformFilteringPress`, also uniform-per-token, collapsed there),
  while at 25–50% all variants were indistinguishable. **Treat ≥75%
  ratios as unsupported.**
- **Chunked-prefill continuation is NOT filtered** (prompt chunks are
  cached in full; the `scheduled == 1` guard excludes them). The research
  doc wanted continuation filtering eventually, but kvpress never
  validated it; v1 stays on the validated path.
- **Cost:** every decode step gathers and scores *all* cached keys for
  *every* layer, plus one GPU→CPU sync per filtered request per step for
  the boolean. Faithful ("scores the full cache") but expensive — the
  known optimization targets are batching across requests/layers and
  incremental anchor maintenance.

---

## 7. File-by-file change map

New:

| File | Contents |
|------|----------|
| `vllm/v1/worker/kv_compression.py` | `keydiff_scores`, `gather_slots`/`scatter_slots`, `compact_request_kv`, `filtering_keep_decision`, `KVCompressionManager` (bind + post-forward driver) |
| `tests/v1/worker/test_kv_compression_standalone.py` | CPU math tests; cross-checks vs kvpress when importable; runnable without a vLLM install (stubs vllm-internal imports) |
| `tests/v1/core/test_kv_compression_scheduler.py` | Scheduler accounting: block savings for A and B, preemption reset |

Modified:

| File | Change |
|------|--------|
| `vllm/config/cache.py` | `kv_compression_algorithm` / `kv_compression_ratio` fields + validator (ratio ∈ (0,1) iff algorithm set; no fp8 cache); both excluded from the compile-graph hash (eager post-forward work, no graph impact) |
| `vllm/config/vllm.py` | Cross-config validation (see §8); force-disables prefix caching |
| `vllm/engine/arg_utils.py` | EngineArgs fields, CLI args, prefix-caching force-off with warning, pass-through to CacheConfig |
| `vllm/v1/request.py` | `Request.num_kv_discarded` (scheduler copy) |
| `vllm/v1/core/sched/scheduler.py` | Reset on preemption; apply reported deltas in `update_from_output` |
| `vllm/v1/core/kv_cache_manager.py` | `allocate_slots`: occupancy = `num_computed_tokens - num_kv_discarded` (single change point; everything downstream — `num_tokens_need_slot`, `get_num_blocks_to_allocate`, `allocate_new_blocks` — flows from it) |
| `vllm/v1/outputs.py` | `ModelRunnerOutput.kv_compression_discarded: dict[str, int] \| None` |
| `vllm/v1/worker/gpu_input_batch.py` | `CachedRequestState.num_kv_discarded` / `.kv_compressed` (worker copies; recreated at defaults when a request is re-added) |
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
2. **Per-head ragged lengths (filtering).** See §6. Would require
   per-layer, per-head seq_lens and kernel awareness — a different
   project.
3. **Continuation (chunked-prefill) filtering.** See §6.
4. **Periodic / pressure-triggered re-compaction.** Mechanism supports
   it (just report another delta); policy left for later.
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
  - `keydiff_scores` ≡ `KeyDiffPress.score` (exact).
  - Compaction ≡ per-head top-k reference on a 3-layer paged cache with
    non-trivial block tables (exact).
  - Filtering decision ≡ the FilteringPress rule on 20/20 randomized
    trials **and exact per-head vote agreement with a real
    `FilteringPress.compress` call**.
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
2. **Majority vote across (layer, head)** for filtering, replacing
   per-head ragged lengths. Safe ≤50% ratio per Phase-1 evidence; not
   validated ≥75%.
3. **Filtering only on single-token decode steps**; continuation chunks
   cached in full.
4. **No tail-block freeing** in v1 of full replacement; savings are
   allocation-deferral, not immediate release.
5. **Compress-once at prefill end** for full replacement (no periodic
   re-compaction).
6. **`n_kept` formulas**: A uses `max(1, int(T·(1−r)))` (kvpress
   `ScorerPress.compress`); B uses `clamp(round(logical·(1−r)), 1, C)`
   (kvpress `FilteringPress`). They intentionally differ because the
   upstream references differ.

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

If a future change ever *did* require touching the backend, it would be
per-head ragged lengths for filtering — that is the one thing the
kernel's single-scalar-bound-per-sequence interface cannot express, and
it is why the majority-vote aggregation exists (§6, §9.2).
