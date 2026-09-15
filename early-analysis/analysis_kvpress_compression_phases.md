# KVPress Compression Analysis

## 1. No continuation prefill compression

Compression is only applied during the **initial prefill**. The gating function `is_prefilling` checks `cache_position[-1] + 1 == q_len`, which is only true when positions start at 0. On a continuation prefill (where positions are offset by the existing cache length), this condition fails and compression is skipped.

This applies to `BasePress`, `ScorerPress`, and all presses built on them. `DecodingPress` and `PrefillDecodingPress` also don't handle continuation prefill — they treat it the same as decoding.

For the evaluation datasets we use (longbench-v2, needle_in_haystack, infinitebench), this gap is irrelevant since they are all single-prefill: one context is prefilled, then questions are answered via decoding.

## 2. Eviction only, no filtering

During decoding, all presses that compress the cache use **eviction**: they periodically re-score the entire cache and remove the lowest-scoring tokens. Any token — including old context tokens — can be evicted.

No press implements **filtering**, where the model decides per-step whether to keep or skip only the current newly generated token without touching past entries.

`QFilterPress`, despite the name, is also eviction-based. The "filter" refers to learned scoring vectors, not a filtering mechanism.
