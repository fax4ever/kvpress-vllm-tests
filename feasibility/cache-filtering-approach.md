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

At each forward pass, when new tokens' K/V tensors are computed:

1. **Score each new token** by comparing its key against recent keys
   already in the cache. Use a similarity metric inspired by KeyDiff:
   tokens whose keys are very similar to existing entries carry
   redundant information.

2. **Decide per token**: if the similarity exceeds a threshold, the
   token is "redundant" — skip writing it to the cache. If it is
   sufficiently different, write it normally via `reshape_and_cache`.

3. **Update metadata**: `seq_lens` reflects only the tokens actually
   cached, not the total tokens processed. The sequence length in the
   attention metadata naturally tracks cache occupancy.

### Scoring with Compression Ratio as Threshold

The keep/skip decision is driven by the compression ratio. For each
new token during decode:

1. **Score** the new token's key against all existing cached keys
   (gathered from the paged cache) using KeyDiff's similarity metric.
2. **Rank** the new token's score against the scores of all context
   tokens — where would this token fall if we ran retroactive KeyDiff
   compression right now?
3. **Decide**: if the token's score places it in the top `ratio`
   fraction (i.e., its probability of being kept is >= the compression
   ratio), cache it. Otherwise, skip it.

This mirrors retroactive compression applied online: instead of
keeping the top 50% after seeing all tokens, we estimate on the fly
whether each new token *would have been* in that top 50%. The
compression ratio directly controls filtering aggressiveness.

Over a long sequence, this converges to roughly `ratio * seq_len`
cached tokens — the same density as retroactive compression, but
achieved incrementally without ever modifying existing cache entries.

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
| **Block allocator** | Must free blocks after compaction, handle partial blocks, update ref counts | **Untouched** — blocks allocated normally, never freed mid-sequence |
| **Block tables** | Must truncate entries for freed blocks (conflicts with append-only design in V1) | **Untouched** — tables only grow |
| **Scheduler** | Must understand compressed sequences, decide when to compress, avoid stalling batches | **Untouched** — sees normal sequences |
| **Prefix caching / CoW** | Must fork shared blocks before compacting (ref_cnt > 1) | **Untouched** — shared blocks are never modified |
| **Cache coherence** | Broken — per-head compaction makes slots hold composite entries | **Preserved** — each slot holds a single token's data |
| **Attention metadata builder** | Must separate logical vs physical seq_len for RoPE, attention bounds, slot mapping | **Minimal change** — track logical position for RoPE only |
| **Cache write path** | Gather + score + scatter back | Gather (read-only for scoring) + filter slot mapping |

### What actually changes in vLLM

The changes are confined to the attention layer's cache write path:

1. **Gather** existing keys from the paged cache for scoring (read-only
   — the same primitive validated in `01_scatter_gather`)
2. **Filter the slot mapping** before calling `reshape_and_cache` —
   tokens that don't pass the score threshold are excluded
3. **Track logical position** separately from cache position for RoPE
   — the logical position always increments, the cache position only
   increments when a token is actually cached

Everything else — block allocator, block tables, scheduler, prefix
caching, continuous batching — works exactly as it does today. The
"hard part" from the full compression roadmap (steps 6–8) disappears
entirely.

## The kvpress Gap

kvpress (NVIDIA's KV cache compression library for Hugging Face
Transformers) implements several compression algorithms, including
KeyDiff. However, all kvpress press types share the same design:

1. Run a full forward pass (prefill)
2. After all tokens are cached, score the entire cache retroactively
3. Keep the top-scoring subset via `torch.gather`

This is fundamentally **batch-retroactive**: the algorithm sees all
tokens before deciding which to keep. It cannot operate in a streaming
fashion, deciding per-token at insertion time.

Cache filtering requires an **online** scoring algorithm:

- The scorer sees only the current token and a window of recent cache
  entries
- The decision (cache or skip) is irrevocable — you cannot reconsider
  past tokens
- The scoring must be fast enough to run at every decode step without
  adding significant latency

No existing kvpress press type supports this mode. A new press type
would be needed — an `OnlineKeyDiffPress` or `StreamingKeyDiffPress`
(implemented as `FilteringPress` wrapping any `ScorerPress`) that:

- Accepts a single new key (or a small batch of new keys during
  chunked prefill)
- Compares against a sliding window of recent cached keys
- Returns a binary keep/skip decision per token
- Maintains minimal state (the window of recent keys for comparison)

This would be a contribution to kvpress itself, extending its
applicability from batch-mode Transformers to streaming serving engines.

## Open Questions

### Scoring Quality

Retroactive scoring has a global view — it can compare all tokens
against each other and keep the most diverse set. Online scoring sees
only the past, not the future. A token that looks redundant now might
have been valuable in hindsight (e.g., a token that is similar to its
neighbors but dissimilar to tokens that arrive later).

The quality gap between retroactive and online scoring needs
experimental validation. It may be acceptable for certain tasks
(long-context QA, summarization) but problematic for others (multi-hop
reasoning where early tokens are referenced later).

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

### Filtering Scope

1. **Prefill**: no. vLLM writes all prompt tokens to the paged cache in a
   single `reshape_and_cache_flash` call — dense tensors, no prior history.
   Filtering is equivalent to retroactive eviction here.
2. **Decoding**: yes. Tokens arrive one at a time, prior tokens are committed
   and append-only — this is where filtering is a distinct operation.
3. **Continuation**: relevant in vLLM, but not implemented in kvpress, so we
   have no baseline to compare against. Phase 1 validation skips it.

The scoring overhead per decode step (one gather + one comparison) must
be kept low to avoid adding latency. But the gather can be limited to
a sliding window of recent keys rather than the full cache, and the
comparison is a single vector operation per head — both are fast
relative to the attention computation itself.

## Probabilistic Compression Ratio

The online approach applies the compression ratio probabilistically,
not exactly. In retroactive compression, you always get precisely
`ratio * seq_len` tokens — you see everything, sort by score, and take
the top fraction. In online filtering, the decision is irrevocable at
insertion time: a token that looks redundant now might have been kept
in hindsight, and vice versa.

Over a long sequence, the actual cache size will **fluctuate around**
the target ratio rather than hitting it exactly. This is acceptable for
a serving engine — the goal is memory savings at a target quality
level, not a precise token count. The statistical properties improve
with sequence length: the law of large numbers means the actual ratio
converges to the target as more tokens are processed.

If tighter control is needed, a simple feedback mechanism could adjust
the threshold dynamically: if the current cache is above the target
density, become slightly more aggressive; if below, become slightly
more conservative.

## Development Strategy

A critical advantage of cache filtering is that the algorithm can be
**validated in kvpress before touching vLLM**. kvpress is a small,
well-structured library — adding a new press type is straightforward
compared to modifying vLLM's paged attention internals. This separation
de-risks the project: the research question (does online scoring
preserve quality?) is answered cheaply before the engineering effort
(vLLM integration) begins.

### Phase 1: Algorithm validation in kvpress

There is already an established contribution path to kvpress: the
`CompressionRatioDecodingPress` class was contributed upstream
([NVIDIA/kvpress#231](https://github.com/NVIDIA/kvpress/pull/231)) and
merged into main. This class is used by our notebooks for evaluating
kvpress baselines. Building `OnlineKeyDiffPress` follows the same
pattern — a new press type added to the same library.

Implement `OnlineKeyDiffPress` as a new kvpress press type. This runs
on Hugging Face Transformers' eager attention path — no paged cache, no
vLLM complexity. The implementation:

- Processes tokens sequentially (simulating decode)
- Scores each new token against all previous tokens
- Applies the compression ratio as a probability threshold
- Returns the filtered K/V cache

Validate against the retroactive `KeyDiffPress` on standard benchmarks:
- **NIAH** (needle-in-a-haystack) — does the filtered cache still find
  the needle?
- **Perplexity** — how does filtering affect language modeling quality?
- **Downstream tasks** — summarization, multi-hop QA, code generation

If the quality gap is unacceptable, the algorithm needs refinement —
and you find out in days, not weeks. Iterating on the scoring function,
the comparison window size, or the threshold strategy is a matter of
changing a few lines in a self-contained kvpress press class and
re-running a notebook. In contrast, the same iteration inside vLLM
would require rebuilding a container image, redeploying to a GPU
cluster, waiting for the job to schedule, and debugging through layers
of paged attention, metadata builders, and slot mappings — a cycle
measured in hours, not minutes. Validating the algorithm in kvpress
first means the expensive vLLM integration only happens once, with a
proven algorithm. If it works, you proceed to vLLM integration with
confidence.

### Phase 2: vLLM integration

With a validated algorithm, the vLLM work is purely engineering:
- Implement the gather primitive for Flash Attention cache layout
- Hook into `do_kv_cache_update` to filter the slot mapping
- Handle the RoPE position tracking
- Test on the focused test suite (basic correctness, attention kernels)

The order matters: kvpress first (10x easier, answers the research
question), vLLM second (harder, but no research risk). If Phase 1
shows poor quality, you can iterate on the algorithm without fighting
vLLM internals. If Phase 1 succeeds, the vLLM integration is a
well-scoped engineering task.

### Contribution

This approach produces two distinct contributions:

1. **`OnlineKeyDiffPress` for kvpress** — extends kvpress from
   batch-retroactive to streaming mode, making it applicable to serving
   engines and real-time inference. This is a contribution to NVIDIA's
   open-source library.

2. **Cache filtering integration for vLLM** — demonstrates that KV
   cache filtering can work within vLLM's paged attention architecture
   without breaking its core invariants. Even as a prototype, this
   validates the feasibility of online cache management in production
   serving engines.

## Comparison with Full Compression

| Aspect | Full compression | Cache filtering |
|--------|-----------------|-----------------|
| Memory savings | Higher — can retroactively remove any token | Lower — can only skip new tokens |
| Integration complexity | Very high — breaks core invariants | Moderate — preserves append-only |
| Scoring quality | Better — global view of all tokens | Potentially worse — local/online view |
| kvpress compatibility | Direct port of existing algorithms | New algorithm needed |
| Latency overhead | Higher — gather + scatter per compression | Moderate — gather for scoring, no scatter back |
| Thesis scope | Steps 3–8 of the roadmap | Primarily steps 3–5, simplified |

## Path Forward

1. **Design the online scoring algorithm** — adapt KeyDiff's
   key-similarity metric for per-token decisions. Define the comparison
   window and threshold.

2. **Implement as a new kvpress press type** — `OnlineKeyDiffPress`
   that works token-by-token, returning keep/skip decisions. Validate
   against the retroactive `KeyDiffPress` on standard benchmarks to
   measure quality loss.

3. **Integrate into vLLM's cache write path** — modify
   `do_kv_cache_update` to filter the slot mapping based on the online
   scorer's decisions. Start with prefill-only filtering on a single
   sequence.

4. **Handle metadata** — track logical vs cache positions for RoPE.
   This is simpler than the full compression case but still requires
   changes to the attention metadata builder.

5. **Benchmark** — measure memory savings vs quality on NIAH and other
   evaluation tasks. Compare against the retroactive approach to
   quantify the trade-off.
