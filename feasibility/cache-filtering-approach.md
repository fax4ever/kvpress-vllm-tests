# Cache Filtering: A Reduced-Scope Alternative to Full Compression

## Motivation

The full KeyDiff compression approach (documented in
`keydiff-vllm-integration.md`) requires a gather-compress-scatter cycle
that fundamentally breaks vLLM's core invariant: **cached tokens are
immutable once written.** The metadata problem alone — separating
logical sequence length from physical cache occupancy — touches the
scheduler, attention metadata builder, and block allocator. Block
management, copy-on-write for shared prefixes, and continuous batching
interactions add further complexity.

An alternative is **cache filtering**: instead of compressing existing
cache entries retroactively, decide at insertion time whether each new
token's K/V is worth caching. Tokens deemed redundant are never written
to the paged cache. The cache still only grows — it just grows more
selectively.

This preserves vLLM's append-only architecture while still reducing
cache memory usage, at the cost of less compression potential (past
tokens cannot be reconsidered).

## How It Works

Cache filtering applies whenever new tokens are added to a sequence
that already has cached context. During the initial prefill, all
prompt tokens are cached normally — there is no prior context to score
against, so the entire prompt is preserved. Filtering activates in
two scenarios:

- **Decode**: tokens arrive one at a time during generation. Each
  token is scored against the full existing cache and independently
  accepted or rejected per head.
- **Continuation** (chunked prefill in vLLM): a batch of new tokens
  arrives for a sequence that already has cached context. Each new
  token in the chunk is scored against the existing cache plus any
  preceding tokens in the same chunk that were accepted.

In kvpress, only decode is supported — kvpress does not implement
continuation. In vLLM, both decode and continuation are relevant, and
the filtering integration must handle both.

At each decode step, the algorithm decides per head whether the new
token is worth caching:

1. **Score all tokens.** The wrapped scorer (KeyDiffPress) scores
   every token in the cache — both existing entries and the new one —
   producing a per-head importance score for each position.

2. **Compute the eviction threshold.** From the total number of tokens
   seen so far and the target compression ratio, compute how many
   tokens should be kept: `n_kept = total_tokens_seen * (1 - ratio)`.
   The threshold is the score of the `n_kept`-th highest token (the
   top-k cutoff).

3. **Decide per head.** Each KV head independently checks whether the
   new token's score is above the threshold. Heads that accept the
   token keep it; heads that reject it mark the position as padding.
   This means different heads can make different decisions about the
   same token — a property that proved essential for quality at
   aggressive compression ratios (see Phase 1 Results).

   **Example.** Suppose 2 heads, 4 cached tokens, 1 new token, and
   `n_kept = 3`:

   ```
            tok0  tok1  tok2  tok3  NEW
   head 0: [ 0.1,  0.8,  0.3,  0.5, 0.2 ]
   head 1: [ 0.4,  0.1,  0.7,  0.3, 0.6 ]
   ```

   Top-3 scores per head: `[0.8, 0.5, 0.3]` and `[0.7, 0.6, 0.4]`.
   The last (smallest) of each top-3 is the threshold: `[0.3, 0.4]`.
   The new token's scores are `[0.2, 0.6]`. Comparing: head 0 rejects
   (0.2 < 0.3), head 1 accepts (0.6 >= 0.4). Each head decides
   independently.

4. **Manage ragged lengths.** Because heads decide independently, the
   valid prefix length can differ across heads. A `PaddedTensor`
   abstraction tracks the per-head valid length. When a head accepts
   the new token but has a gap (from earlier rejections by that head),
   the token is copied to the first padding slot to keep valid entries
   packed at the front. When all heads have padding at the last
   position, the backing tensor is shrunk.

The scoring step mirrors retroactive compression applied online:
instead of keeping the top 50% after seeing all tokens, the algorithm
checks at each step whether the new token *would have survived*
retroactive compression given everything seen so far. The compression
ratio is the only parameter — no separate threshold needs tuning.

Unlike retroactive compression, which achieves exactly
`ratio * seq_len` kept tokens by sorting all scores and taking the
top fraction, online filtering achieves the target compression ratio
**in expectation**. Each keep/skip decision is made independently
based on the current cache state, so the actual number of cached
tokens fluctuates around the target rather than hitting it precisely.
Over a long sequence, the law of large numbers ensures convergence:
the realised compression ratio approaches the target as more tokens
are processed. This statistical property is sufficient for a serving
engine, where the goal is memory savings at a target quality level,
not an exact token count.

## Where It Fits in vLLM

The current cache write path:

```text
Attention.forward()
  ├── do_kv_cache_update(key, value, kv_cache, slot_mapping)
  │     ├── split_kv_cache(kv_cache) → key_cache, value_cache
  │     └── reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
  └── chunked_prefill_paged_decode(query, key, value, ...)
```

Cache filtering modifies the cache write step:

```text
Attention.forward()
  ├── do_kv_cache_update(key, value, kv_cache, slot_mapping)
  │     ├── split_kv_cache(kv_cache) → key_cache, value_cache
  │     ├── gather_recent_keys(key_cache, block_table, ...)          ← NEW
  │     ├── score_new_tokens(key, gathered_keys, ...)                ← NEW
  │     ├── filter slot_mapping to keep only non-redundant tokens    ← NEW
  │     └── reshape_and_cache(key, value, ..., filtered_slot_mapping)
  └── chunked_prefill_paged_decode(query, key, value, ...)
```

The **gather** operation is still required: to decide whether a new
token is redundant, we must compare its key against existing cached
keys. Those keys live in the paged cache, scattered across physical
blocks, so they must be gathered into a dense buffer for comparison.
This is the same gather primitive described in
`keydiff-vllm-integration.md` — reading from the paged cache via block
table indexing into a contiguous tensor.

The difference from full compression is that we only gather (read) —
we never scatter compressed data back. The gather serves the scoring
step, not a rewrite cycle. After scoring, `reshape_and_cache` is
driven by `slot_mapping`. By filtering the slot mapping (and the
corresponding entries in the dense K/V tensors), we control which new
tokens get written. Tokens excluded from the filtered mapping are
simply never cached. No scatter back, no block freeing.

## What This Avoids

The append-only nature of cache filtering means that entire layers of
vLLM's architecture are **completely untouched**. The full compression
approach requires changes to the block allocator, block tables,
scheduler, and metadata builder. Cache filtering requires none of that
— these subsystems see normal sequences that just happen to grow
slower than expected.

### vLLM subsystems: what changes and what doesn't

| Subsystem | Full compression | Cache filtering |
|-----------|-----------------|-----------------|
| **Block allocator** | Must free blocks after compaction, handle partial blocks, update ref counts | **Needs change** — must allocate based on cached tokens, not computed tokens (see note below) |
| **Block tables** | Must truncate entries for freed blocks (conflicts with append-only design in V1) | **Untouched** — tables only grow |
| **Scheduler** | Must understand compressed sequences, decide when to compress, avoid stalling batches | **Needs change** — `num_computed_tokens` conflates computation cursor and cache occupancy (see note below) |
| **Prefix caching / CoW** | Must fork shared blocks before compacting (ref_cnt > 1) | **Untouched** — shared blocks are never modified |
| **Cache coherence** | Broken — per-head compaction makes slots hold composite entries | **Preserved** — each slot holds a single token's data |
| **Attention metadata builder** | Must separate logical vs physical seq_len for RoPE, attention bounds, slot mapping | **Minimal change** — track logical position for RoPE only |
| **Cache write path** | Gather + score + scatter back | Gather (read-only for scoring) + filter slot mapping |

### What actually changes in vLLM

The changes involve the cache write path **and** the allocation chain:

1. **Gather** existing keys from the paged cache for scoring (read-only
   — the same primitive validated in `01_scatter_gather`)
2. **Filter the slot mapping** before calling `reshape_and_cache` —
   tokens that don't pass the score threshold are excluded
3. **Track logical position** separately from cache position for RoPE
   — the logical position always increments, the cache position only
   increments when a token is actually cached
4. **Separate computation cursor from cache occupancy.** Currently
   `request.num_computed_tokens` serves two roles: (a) where to resume
   the forward pass, and (b) how many cache slots are occupied. The
   block allocator (`KVCacheManager.allocate_slots`,
   `kv_cache_manager.py:218`) uses it to compute
   `ceil(total_tokens / block_size)` blocks to allocate. With filtering,
   these two roles must diverge — the computation cursor must always
   increment (the model processed the token), but the cache occupancy
   should reflect only the tokens actually written to cache. Without
   this separation, blocks are allocated for all computed tokens
   including filtered ones, and no memory is saved.

This is simpler than the full compression approach (no scatter back,
no block freeing, no copy-on-write on shared prefixes), but the block
allocator and scheduler are **not** untouched as originally claimed.
The allocation chain must be aware of filtering to achieve actual
memory savings.

## The kvpress Gap (resolved)

Prior to this work, all kvpress press types shared the same design:

1. Run a full forward pass (prefill)
2. After all tokens are cached, score the entire cache retroactively
3. Keep the top-scoring subset via `torch.gather`

This is fundamentally **batch-retroactive**: the algorithm sees all
tokens before deciding which to keep. It cannot operate in a streaming
fashion, deciding per-token at insertion time. Cache filtering requires
an **online** scoring algorithm where the decision is irrevocable and
fast enough to run at every decode step.

`FilteringPress` (contributed as PR #257) closes this gap. It is a
`DecodingPress` subclass that wraps any `ScorerPress` and makes
online keep/skip decisions during decode. The key design choices:

- **Reuses existing scorers.** `FilteringPress` does not implement its
  own scoring — it delegates to the wrapped press (e.g. `KeyDiffPress`).
  This means any future scorer added to kvpress automatically works
  with filtering.
- **Scores the full cache, not a sliding window.** The original design
  hypothesised a sliding window of recent keys for efficiency. The
  implementation scores all cached tokens at each step, which is
  simpler and produces better decisions. The overhead is acceptable
  because the scorer runs once per decode step and the cache is
  already bounded by the compression ratio.
- **Per-head decisions with `PaddedTensor`.** Each head decides
  independently, requiring a data structure to track ragged per-head
  valid lengths. `PaddedTensor` manages this with minimal overhead.
- **`UniformFilteringPress` as simpler alternative.** A majority-vote
  variant was also implemented but proved unsuitable at aggressive
  compression ratios (see Phase 1 Results).

## Open Questions

### Scoring Quality

Retroactive scoring has a global view — it can compare all tokens
against each other and keep the most diverse set. Online scoring sees
only the past, not the future. A token that looks redundant now might
have been valuable in hindsight (e.g., a token that is similar to its
neighbors but dissimilar to tokens that arrive later).

**Update (Phase 1 results):** NIAH experiments show no measurable
quality gap between online filtering and retroactive eviction at
compression ratios up to 50%. At 75%, both approaches degrade
similarly — the gap is in the compression aggressiveness, not the
online vs retroactive distinction. The concern about hindsight does
not materialise at practical compression levels, at least on the NIAH
retrieval task. Validation on other tasks (multi-hop reasoning, code
generation) remains future work.

### RoPE Positions

When a token is skipped (not cached), the next token's RoPE position
must still reflect its logical position in the original sequence, not
its cache slot index. Otherwise the positional encodings would be
wrong.

This requires tracking the logical position separately from the cache
write position. However, this is simpler than the full compression
case: the logical position always increments by 1, regardless of
whether the token was cached. The mapping is:

- `logical_position` — always `previous + 1`, used for RoPE
- `cache_position` — increments only when a token is actually cached,
  used for slot mapping

Since RoPE is applied to the key *before* it enters the cache (the key
already carries its positional encoding from the model's forward pass),
skipping a token does not affect the RoPE values of other cached
tokens. The skipped token's position is simply absent from the cache,
which is equivalent to the attention kernel seeing a shorter sequence.

### Threshold Tuning

The keep/skip decision depends on a similarity threshold. Too
aggressive (high threshold, skip many tokens) loses important context.
Too conservative (low threshold, skip few tokens) provides little
memory savings.

The threshold may need to be:
- **Adaptive per head** — different attention heads attend to different
  aspects; a head that tracks syntactic structure might need more tokens
  than one that tracks semantic content
- **Adaptive over time** — early tokens in a sequence may be more
  important (establishing context) than later tokens in a long
  generation
- **Configurable per deployment** — different use cases (chatbot vs
  document QA vs code generation) may tolerate different compression
  levels

**Update (Phase 1 results):** The `FilteringPress` implementation
uses the compression ratio directly as the threshold — it checks
whether the new token's score would place it above the top-k cutoff
if retroactive compression were applied right now. This approach
avoids the need for a separate tunable threshold parameter: the
compression ratio is the only knob, and it maps directly to the
user's intent (target cache density). Per-head decisions proved
essential — the majority-vote approach in `UniformFilteringPress`
collapses at aggressive ratios.

### Filtering Scope

1. **Prefill**: no. vLLM writes all prompt tokens to the paged cache
   in a single `reshape_and_cache_flash` call — dense tensors, no
   prior history. Filtering is equivalent to retroactive eviction here.
2. **Decoding**: yes. Tokens arrive one at a time, prior tokens are
   committed and append-only — this is where filtering is a distinct
   operation. Validated in Phase 1 via kvpress.
3. **Continuation** (chunked prefill): yes. In vLLM, a batch of new
   tokens can arrive for a sequence that already has cached context.
   The new tokens should be scored against the existing cache (and
   accepted predecessors in the same chunk) and filtered before being
   written. This is not implemented in kvpress — kvpress does not
   support continuation — so there is no Phase 1 baseline. The vLLM
   integration must handle this case directly.

**Update (Phase 1 results):** The implementation scores the full
cache at each step rather than using a sliding window. The overhead is
acceptable because the cache is already bounded by the compression
ratio and the scorer runs once per step — fast relative to the
attention computation itself.

## Development Strategy

A critical advantage of cache filtering is that the algorithm can be
**validated in kvpress before touching vLLM**. kvpress is a small,
well-structured library — adding a new press type is straightforward
compared to modifying vLLM's paged attention internals. This separation
de-risks the project: the research question (does online scoring
preserve quality?) is answered cheaply before the engineering effort
(vLLM integration) begins.

### Phase 1: Algorithm validation in kvpress (completed)

Following the established contribution path to kvpress — the
`CompressionRatioDecodingPress` class was contributed upstream
([NVIDIA/kvpress#231](https://github.com/NVIDIA/kvpress/pull/231)) and
merged into main — `FilteringPress` was implemented as a new kvpress
press type (PR #257). It runs on Hugging Face Transformers' eager
attention path, with no paged cache or vLLM complexity, which allowed
rapid iteration on the algorithm before committing to the harder
vLLM integration.

The algorithm was validated against the retroactive `KeyDiffPress` on
the NIAH (needle-in-a-haystack) benchmark. Perplexity and downstream
task evaluation (summarization, multi-hop QA, code generation) remain
future work, but NIAH was sufficient to answer the core research
question: does online scoring preserve retrieval quality?

### Phase 1 Results

Phase 1 was implemented as PR #257 to NVIDIA/kvpress (commit `635d923`
on the `filtering` branch). Two new press types were contributed:

- **`FilteringPress`** — per-head online keep/skip decisions. Each
  head independently scores all tokens (including the new one) and
  rejects it if its score falls below the eviction threshold at the
  target compression ratio. Rejected heads mark the position as
  padding via a `PaddedTensor` abstraction; accepted heads that find
  an earlier padding slot move the token there to keep valid tokens
  packed. Has a `fill_padding` option to zero out stale data in
  padding positions. Tested in two variants: zero-filled padding
  (`filtering_zerofill_keydiff`) and stale padding
  (`filtering_stale_keydiff`).

- **`UniformFilteringPress`** — simpler all-or-nothing variant. Each
  head votes independently, then a majority vote produces a single
  keep/remove decision for the token. No ragged lengths, no
  `PaddedTensor` needed.

Both are `DecodingPress` subclasses — they are no-ops during prefill
and only filter during decode, where tokens arrive one at a time and
the cache is append-only. All configurations use `KeyDiffPress` as
the underlying scorer, wrapped with `PrefillDecodingPress` to apply
retroactive `KeyDiffPress` compression during prefill.

The NIAH benchmark (`notebooks/02_kvpress_niah.ipynb`) tested four
press configurations on Qwen3-8B at compression ratios 0%, 25%, 50%,
75% across context lengths 4096 and 8192, with needle depths at 0%,
25%, 50%, 75%, and 100%:

| Press configuration | 0% ratio | 25% ratio | 50% ratio | 75% ratio |
|---------------------|----------|-----------|-----------|-----------|
| Retroactive (`CompressionRatioDecodingPress`) | 0.72 | 0.73 | 0.74 | 0.64 |
| `FilteringPress` (zero-fill) | 0.72 | 0.73 | 0.74 | 0.68 |
| `FilteringPress` (stale) | 0.72 | 0.73 | 0.74 | 0.67 |
| `UniformFilteringPress` | 0.72 | 0.73 | 0.74 | **0.19** |

*(Average ROUGE-L F-score across all needle depths and context
lengths. Higher is better; uncompressed baseline is 0.72.)*

#### Key observations

**Filtering matches retroactive compression up to 50%.** At
compression ratios of 0%, 25%, and 50%, all four approaches produce
essentially identical ROUGE-L scores. There is no measurable quality
gap between online filtering and retroactive eviction at these
practical compression levels. This is the central result: the concern
that online scoring (seeing only past tokens, not future ones) would
degrade quality relative to retroactive scoring (global view) does not
materialise at moderate compression ratios.

**At 75%, per-head `FilteringPress` degrades gracefully.** Both the
zero-fill and stale variants show quality drops at certain needle
depths — particularly depth 25% and 50% at context length 8192, where
ROUGE-L falls to 0.34–0.67. This is comparable to the retroactive
approach, which shows similar depth-dependent drops (0.41–0.74 at the
same points). The degradation pattern is consistent: aggressive
compression occasionally drops the needle tokens, and the effect is
worse when the needle sits in a region of high textual similarity
where KeyDiff is more likely to score it as redundant.

**`UniformFilteringPress` catastrophically fails at 75%.** The
majority-vote mechanism produces degenerate repetitive outputs at
aggressive compression ratios: the model generates sequences like "to
to to to to..." with ROUGE-L scores as low as 0.13. The all-or-nothing
decision is too coarse — when a majority of heads reject a token, the
model loses it entirely, and the cumulative effect over many decode
steps causes the model to lose coherence. This makes
`UniformFilteringPress` unsuitable for compression ratios above 50%.

**Zero-fill vs stale padding makes negligible difference.** The two
`FilteringPress` variants (zeroing out padding positions vs leaving
stale data) produce nearly identical results across all configurations.
This suggests that the attention mechanism is robust to small amounts
of stale data in padding positions, likely because the valid mask
prevents these positions from contributing meaningfully to attention
scores.

#### Implications for vLLM integration

These results validate cache filtering as a viable strategy for vLLM:

1. **The algorithm works.** Online keep/skip decisions preserve quality
   comparable to retroactive compression at practical compression
   ratios (up to 50%). The quality gap is negligible where it matters
   most — moderate compression that balances memory savings with output
   quality.

2. **Per-head decisions are essential.** `FilteringPress` (per-head)
   significantly outperforms `UniformFilteringPress` (majority vote)
   at aggressive compression. For vLLM integration, the per-head
   approach is the right one to pursue, despite the added complexity
   of ragged per-head lengths.

3. **The full-compression roadmap is unnecessary.** The original plan
   (`keydiff-vllm-integration.md`) proposed a gather-compress-scatter
   cycle that breaks vLLM's append-only invariant and requires
   invasive changes to the block allocator, scheduler, block tables,
   and metadata builder (steps 6–8 of that roadmap). Since filtering
   achieves comparable quality without modifying existing cache
   entries, those hard integration problems disappear entirely.

4. **50% is the practical ceiling.** Both filtering and retroactive
   compression show quality degradation at 75%. A vLLM integration
   that targets 25–50% compression would operate in the regime where
   filtering is indistinguishable from retroactive compression — the
   stronger guarantee.

5. **The `fill_padding` optimisation is safe.** Since zero-fill and
   stale variants are equivalent in quality, the vLLM implementation
   can skip the zero-fill step, avoiding the overhead of writing zeros
   to cache positions that the attention mask already excludes.

### Phase 2: vLLM integration

With Phase 1 validating that online filtering preserves quality, the
vLLM integration is purely engineering — no research risk remains. The
algorithm is proven; the task is adapting it to vLLM's paged attention
architecture.

The original full-compression roadmap (`keydiff-vllm-integration.md`)
identified steps 3–5 as warm-up exercises and steps 6–8 as the hard
system-level integration. With filtering, steps 6–8 largely disappear:
there is no metadata separation problem (logical vs physical sequence
length), no block freeing after compaction, no copy-on-write for
shared prefix blocks, and no scheduler changes. The append-only
invariant is preserved, so vLLM's block allocator, scheduler, and
prefix caching all work unchanged.

The warm-up exercises from the original roadmap remain directly useful,
however — they build familiarity with vLLM's Flash Attention cache
layout and the gather primitive, both of which filtering still needs.

#### Step 1: Gather on Flash Attention cache layout

Port the gather function from the classic paged attention layout
(validated in `vllm-experiments/01_scatter_gather`) to Flash Attention
2's layout: `[num_blocks, block_size, num_kv_heads, head_size]`. This
is the same 4D shape for both keys and values, simpler than the
classic layout (5D keys, 4D values with different dimension ordering).

The gather is read-only: given a sequence's block table and sequence
length, read all cached keys into a dense
`[seq_len, num_kv_heads, head_size]` tensor. The scatter direction
uses `reshape_and_cache_flash` (already exists). Validate with a
round-trip test: scatter known data, gather it back, verify numerical
equality.

This step builds confidence with vLLM's cache internals and produces
the gather primitive that filtering needs for scoring.

#### Step 2: End-to-end scoring

Combine the gather primitive with KeyDiff's key-similarity scoring.
Given a sequence's cached keys (gathered from the paged cache) and a
new token's key, compute the score and make a keep/skip decision.

Validate against kvpress's `FilteringPress` on the same input: given
the same keys and the same scorer, the vLLM-side scoring should
produce the same decision. This ensures the scoring logic is correctly
ported before integrating it into the forward pass.

#### Step 3: FilteringPress integration

Insert filtering into vLLM's attention layer. The insertion point is
between `do_kv_cache_update` and `chunked_prefill_paged_decode` in
`Attention.forward()`:

```text
Attention.forward()
  ├── do_kv_cache_update(key, value, kv_cache, slot_mapping)
  │     ├── split_kv_cache(kv_cache) → key_cache, value_cache
  │     ├── gather_cached_keys(key_cache, block_table, ...)     ← NEW
  │     ├── score_and_filter(key, gathered_keys, ...)           ← NEW
  │     └── reshape_and_cache_flash(key, value, ..., filtered_slot_mapping)
  └── chunked_prefill_paged_decode(query, key, value, ...)
```

During decode, for each new token:
1. **Gather** existing cached keys (read-only, for scoring)
2. **Score** the new token against all cached keys using KeyDiff
3. **Filter**: if the token's score is below the eviction threshold,
   exclude it from the slot mapping so it is never written to the
   cache
4. **Track logical position** for RoPE — the logical position always
   increments even when a token is skipped, because RoPE is baked
   into the key before it enters the cache

During prefill, filtering is a no-op — all prompt tokens are cached
normally, matching the kvpress implementation where `FilteringPress`
is a `DecodingPress` subclass.

#### Step 4: NIAH benchmark

Run the same Needle-in-a-Haystack benchmark used for kvpress
evaluation, now on the vLLM fork with filtering enabled. Compare:
- vLLM with filtering vs vLLM baseline (uncompressed)
- vLLM with filtering vs kvpress `FilteringPress` results

This closes the loop: if the vLLM results match kvpress results at the
same compression ratios, the integration is validated end-to-end.

### Contribution

This approach produces two distinct contributions:

1. **`FilteringPress` for kvpress** (completed) — extends kvpress from
   batch-retroactive to streaming mode, making it applicable to serving
   engines and real-time inference. Contributed as PR #257 to
   NVIDIA/kvpress. Includes `FilteringPress` (per-head decisions with
   `PaddedTensor` for ragged lengths) and `UniformFilteringPress`
   (majority-vote all-or-nothing decisions). NIAH evaluation shows
   per-head filtering matches retroactive compression up to 50%
   compression ratio.

2. **Cache filtering integration for vLLM** (in progress) —
   demonstrates that KV cache filtering can work within vLLM's paged
   attention architecture without breaking its core invariants. Unlike
   the full-compression approach, filtering preserves vLLM's
   append-only cache semantics, avoiding changes to the block
   allocator, scheduler, and metadata builder.

## Comparison with Full Compression

| Aspect | Full compression | Cache filtering |
|--------|-----------------|-----------------|
| Memory savings | Higher — can retroactively remove any token | Lower — can only skip new tokens |
| Integration complexity | Very high — breaks core invariants | Moderate — preserves append-only |
| Scoring quality | Global view of all tokens | Equivalent up to 50% ratio (validated on NIAH); both degrade at 75% |
| kvpress compatibility | Direct port of existing algorithms | `FilteringPress` contributed (PR #257) |
| Latency overhead | Higher — gather + scatter per compression | Moderate — gather for scoring, no scatter back |
| Thesis scope | Steps 3–8 of the roadmap | Primarily steps 3–5, simplified |

## Path Forward

### Completed

1. **Online scoring algorithm** — `FilteringPress` adapts KeyDiff's
   key-similarity metric for per-token keep/skip decisions during
   decode. The scorer operates on the full cache (not a sliding
   window), comparing the new token against all existing entries.
   The compression ratio drives the eviction threshold.

2. **kvpress implementation and validation** — `FilteringPress` and
   `UniformFilteringPress` contributed as PR #257 to NVIDIA/kvpress.
   NIAH evaluation confirms that per-head filtering matches
   retroactive compression quality up to 50% compression ratio.
   `UniformFilteringPress` (majority vote) fails at aggressive
   ratios and is not recommended for vLLM integration.

### Next

3. **Gather on Flash Attention cache layout** — port the gather
   primitive to the `[num_blocks, block_size, num_kv_heads,
   head_size]` layout used on NVIDIA GPUs. Validate with round-trip
   tests against `reshape_and_cache_flash`.

4. **End-to-end scoring on paged cache** — combine gather with
   KeyDiff scoring. Verify that decisions match kvpress on the same
   inputs.

5. **FilteringPress in vLLM** — filter the slot mapping before
   `reshape_and_cache_flash` in the attention forward pass. Track
   logical positions for RoPE separately from cache positions.

6. **NIAH benchmark on vLLM** — compare vLLM with filtering against
   both the uncompressed vLLM baseline and the kvpress filtering
   results to validate end-to-end.
