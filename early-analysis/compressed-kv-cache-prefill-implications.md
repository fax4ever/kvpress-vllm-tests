# Implications of a compressed KV cache for prefill

This note describes what changes in prefill when the KV cache is compressed.

It is about prefill only. The companion document covers compressed-cache
decode.

It describes the compressed-cache prefill changes implemented on the
`key-diff` branch. It is not a description of the current upstream
implementation on `main`.

## Recap: two-phase prefill

Standard prefill attention has two sequential phases:

1. **Context phase** — query tokens attend to previously cached KV entries
   (the prefix from earlier iterations)
2. **Request phase** — query tokens attend to their own K/V from the
   current forward pass (with causal masking)

These two phases have different KV sources:

- Context phase reads from the paged KV cache via the block table
- Request phase reads from the raw K/V tensors

## What changes with a compressed cache

Compression affects **only the context phase**.

In the standard case, the context phase iterates over all previously cached
tokens. Conceptually:

```text
prefix: [1, 1, 1, 1, 1]    →  context_len = 5
```

With a compressed cache, only some prefix tokens are kept:

```text
prefix: [1, 1, 0, 1, 0]    →  compacted_ctx_len = 3
```

The dropped tokens are not in the cache at all. The kept tokens are packed
contiguously in the paged KV cache, so the first 3 cache slots contain the
surviving entries.

The request phase is completely unaffected. The new query tokens still
attend to each other using the raw K/V tensors with the standard causal
mask. Compression does not touch this phase because the new tokens have not
been compressed — they are being processed for the first time.

## Why the two-phase structure makes this clean

The existing kernel already separates context and request into two distinct
loops with different KV sources. This means compressed-cache prefill only
needs to adjust the first loop's bound and masks. The second loop does not
need any change.

This is a structural advantage over a single-loop design where cached and
new KV would be interleaved.

## What changes in the kernel

The change is small: a new variable `cur_batch_cache_len` replaces
`cur_batch_ctx_len` in the context loop.

In the standard case:

```python
cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len
# context loop iterates over cur_batch_ctx_len
```

With compressed cache:

```python
cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len
if USE_COMPACTED_CONTEXT:
    cur_batch_cache_len = tl.load(B_Compacted_Ctx_Len + cur_batch)
else:
    cur_batch_cache_len = cur_batch_ctx_len
```

Then everywhere in the context phase, `cur_batch_ctx_len` is replaced by
`cur_batch_cache_len`:

- the context loop bound: `for start_n in tl.range(0, cur_batch_cache_len, ...)`
- the boundary masks: `(start_n + offs_bs_n[...]) < cur_batch_cache_len`
- the conditional masking for tile loads

The request phase continues to use `cur_batch_query_len` as before.

## How `cur_batch_cache_len` is computed

A host-side helper `_build_compacted_context_lens` counts the kept prefix
tokens before launching the kernel:

```python
def _build_compacted_context_lens(token_presence_bitmap, context_lens):
    # token_presence_bitmap: [num_seqs, num_kv_heads, max_ctx_len]
    # context_lens: [num_seqs]
    # returns: [num_seqs] with the number of kept prefix tokens
```

It masks the bitmap by each sequence's actual context length, counts the
number of `1` entries, and produces a per-sequence compacted context length.

A validation check ensures all KV heads keep the same number of prefix
tokens per sequence. This is required because the context loop iterates
with a single scalar bound, not a per-head bound.

## Detailed walkthrough of `_build_compacted_context_lens`

Consider a batch of 2 sequences, 2 KV heads, and a max context of 5 slots:

```python
token_presence_bitmap = [
    [[1, 1, 0, 1, 0],   # seq 0, kv_head 0
     [1, 1, 0, 1, 0]],  # seq 0, kv_head 1
    [[1, 0, 1, 1, 1],   # seq 1, kv_head 0
     [1, 0, 1, 1, 1]],  # seq 1, kv_head 1
]
context_lens = [4, 3]
```

Sequence 0 has a logical context of 4 tokens (positions 0–3), and sequence
1 has a logical context of 3 tokens (positions 0–2). The bitmap has 5
columns, but positions beyond the context length are irrelevant.

**Step 1: build a position index and mask out-of-range positions.**

```python
positions     = [0, 1, 2, 3, 4]

# seq 0, context_len=4: positions < 4
valid_pos[0]  = [T, T, T, T, F]

# seq 1, context_len=3: positions < 3
valid_pos[1]  = [T, T, T, F, F]
```

**Step 2: combine with the bitmap to get a present mask.**

```python
# seq 0, head 0: bitmap & valid_pos
present[0][0] = [1, 1, 0, 1, 0] & [T, T, T, T, F] = [T, T, F, T, F]

# seq 0, head 1:
present[0][1] = [1, 1, 0, 1, 0] & [T, T, T, T, F] = [T, T, F, T, F]

# seq 1, head 0: bitmap & valid_pos
present[1][0] = [1, 0, 1, 1, 1] & [T, T, T, F, F] = [T, F, T, F, F]

# seq 1, head 1:
present[1][1] = [1, 0, 1, 1, 1] & [T, T, T, F, F] = [T, F, T, F, F]
```

Note how position 3 in seq 1 has a `1` in the bitmap but is masked out
because `context_len=3` means only positions 0–2 are valid. The bitmap
may be wider than the actual context; the `valid_positions` mask prevents
overcounting.

**Step 3: count kept tokens per (sequence, head).**

```python
kept_counts = present.sum(dim=-1)
# seq 0: [3, 3]   (both heads keep 3)
# seq 1: [2, 2]   (both heads keep 2)
```

**Step 4: take head 0's count and verify all heads agree.**

```python
compacted_ctx_lens = kept_counts[:, 0]  # [3, 2]
# check: kept_counts == compacted_ctx_lens[:, None].expand_as(kept_counts)
# [3, 3] == [3, 3] ✓
# [2, 2] == [2, 2] ✓
```

If the heads disagreed (e.g., head 0 kept 3 but head 1 kept 2), the check
would fail with a `ValueError`.

**Result:** `compacted_ctx_lens = [3, 2]`. These two integers are all that
the kernel receives. The bitmap itself is never passed to the GPU kernel.

The kernel then uses `3` instead of `4` as the context loop bound for
sequence 0, and `2` instead of `3` for sequence 1. This means fewer cache
tiles are loaded, fewer dot products are computed, and fewer softmax
contributions are accumulated — the kept tokens are already packed into the
first `compacted_ctx_len` slots of the paged cache.

## What remains true

The attention computation itself is unchanged:

```text
Attention(Q, K_kept, V_kept) = softmax(Q · K_kept^T) · V_kept
```

The difference is in which keys and values are present in the cache, not
in the formula.

Specifically:

- the online softmax accumulation is identical
- the dot-product tile structure is identical
- the block-table addressing is identical (it still maps logical tile
  positions to physical block IDs)
- the request phase (causal self-attention over new tokens) is identical

## What does not need to change

The request phase does not need any modification because:

- it reads from raw K/V tensors, not from the cache
- it applies causal masking among the new tokens
- compression has no effect on tokens that have not been compressed

The kernel grid does not change either. The grid is:

```python
grid = (batch, head, triton.cdiv(max_input_len, BLOCK_M))
```

This depends on the query length, which is unaffected by cache compression.

## What must change

Even though the change is small, it is not "no code change". At minimum:

- the context loop must use the compacted prefix length instead of the
  original context length
- all boundary masks in the context loop must use that compacted length
- the compacted length must be precomputed on the host and passed to the
  kernel as an additional parameter

## Interaction with the entry point

The `token_presence_bitmap` is passed through the call chain:

```text
chunked_prefill_paged_decode(token_presence_bitmap=...)
  └─ context_attention_fwd(token_presence_bitmap=...)
       └─ _build_compacted_context_lens(token_presence_bitmap, context_lens)
       └─ _fwd_kernel(B_Compacted_Ctx_Len=..., USE_COMPACTED_CONTEXT=True)
```

The same `chunked_prefill_paged_decode` call also handles the decode path
for single-token sequences in the batch. The `token_presence_bitmap` is
used there too, via a parallel `_build_compacted_seq_lens` helper that
computes compacted lengths for decode sequences.

When a `token_presence_bitmap` is present, the ROCm custom paged attention
kernel is disabled and the Triton fallback is used for decode. This is
because the custom HIP kernel does not support compacted sequence lengths.

## Unsupported features

Compressed-cache prefill does not currently support:

- ALiBi (position-dependent biases would need the original logical
  positions of kept tokens)
- sliding-window masking (the window bound relies on the original
  position of each token)

Both raise `NotImplementedError` if combined with a `token_presence_bitmap`.

## Comparison with compressed-cache decode

| Aspect | Compressed prefill | Compressed decode |
|--------|-------------------|-------------------|
| What changes | Context loop bound and masks | Main loop bound and masks |
| What is unchanged | Request phase (new tokens) | N/A (decode has no request phase) |
| Compacted length variable | `cur_batch_cache_len` | `compacted_seq_len` |
| Host helper | `_build_compacted_context_lens` | `_build_compacted_seq_lens` |
| Kernel parameter | `B_Compacted_Ctx_Len` | `compacted_seq_lens_ptr` |
| Compile-time flag | `USE_COMPACTED_CONTEXT` | `USE_COMPACTED_SEQ_LENS` |

The two changes are structurally symmetric: both replace an iteration
bound with a compacted variant. Prefill is slightly more involved because
the compacted length must be separated from the query length
(`ctx_len = seq_len - query_len`), whereas decode can directly compact
`seq_len`.

## Summary

Compressed-cache prefill changes the context seen by the first phase of
the prefill kernel: instead of attending to all previously cached prefix
tokens, the query attends only to the kept subset.

The request phase — where new tokens attend to each other — is completely
unaffected by compression.

The implementation adds a single new length variable (`cur_batch_cache_len`)
that replaces `cur_batch_ctx_len` in the context loop, with a host-side
helper to count kept tokens from the presence bitmap. The rest of the
kernel — the online softmax, the block-table addressing, the request
loop, and the output shape — remains unchanged.
