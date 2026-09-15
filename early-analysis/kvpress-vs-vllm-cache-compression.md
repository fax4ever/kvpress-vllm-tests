# Why kvpress can shrink tensors and vLLM needs a compacted length

This note discusses the structural differences between how kvpress and
vLLM handle KV cache compression, and why each approach works for its
respective framework.

## How kvpress compresses the cache

kvpress is built on HuggingFace Transformers. After each attention layer
during prefill, a forward hook extracts the K/V tensors from the cache,
scores each token, keeps the top-scoring ones via `torch.gather`, and
writes the shorter tensors back:

```python
n_kept = int(k_len * (1 - self.compression_ratio))
indices = scores.topk(n_kept, dim=-1).indices
keys = keys.gather(2, indices).contiguous()
values = values.gather(2, indices).contiguous()
```

After this, the K/V tensors physically have shape
`[batch, heads, n_kept, head_dim]` instead of
`[batch, heads, seq_len, head_dim]`. The dropped tokens are gone from
memory.

## Why shrinking works in kvpress

The cache is only used in two situations, and neither requires a causal
relationship between the cached entries themselves:

1. **Prefill context** — the cache stores previously seen tokens. Attention
   over these tokens has no causal mask, because all context tokens are in
   the past relative to every query token. Removing some entries and keeping
   the rest produces a valid, smaller context. The remaining tokens form a
   subset that attention can operate over without any masking issues.

2. **Decode** — the query is a single new token attending to all cached
   past tokens. Again, there is no masking among the cached entries. A
   shorter cache simply means fewer past tokens contribute to the
   attention-weighted sum.

In neither case is there a causal relationship *between* cached entries
that would break if some were removed and the rest shifted. The causal
mask only exists in the prefill request phase (new tokens attending to
each other), and kvpress does not compress that — the hook runs *after*
attention has already been computed for the current layer.

So when HuggingFace Transformers later runs attention (either during
subsequent prefill layers or during decode), it simply operates over
whatever K/V tensors it receives. The tensor shape is the truth — no
separate compacted length is needed.

## Why vLLM needs a compacted length

vLLM uses a paged KV cache: a pre-allocated pool of physical blocks,
managed through a block table. The cache is not a simple tensor whose
shape can be inspected — it is a fixed structure with:

- a pre-allocated pool of physical blocks
- a block table mapping logical blocks to physical blocks
- sequence length metadata (`seq_lens`) that the kernel uses as loop bounds

After compression, the kept tokens are packed contiguously into the
first `compacted_ctx_len` slots — there are no gaps or fragmentation.
However, unlike a regular tensor, the paged cache does not carry its
own "how many entries are valid" information in its shape. The kernel
relies on metadata for that. Since `seq_lens` still reflects the
original uncompressed length, a separate compacted length is needed
to tell the kernel the correct, smaller bound.

## The compacted length as an interface

In the `key-diff` branch, the compacted length is derived from the
`token_presence_bitmap` by counting kept tokens. The bitmap is consumed
on the host, and the kernel receives just the count:

- prefill context phase: `cur_batch_cache_len` replaces `cur_batch_ctx_len`
- decode: `compacted_seq_len` replaces `seq_len`

This is the minimal adaptation needed: the kernel iterates over fewer
cache slots, and the boundary masks use the compacted length. The
attention formula, the block-table addressing, and the online softmax
remain unchanged.

## Why contiguous compaction preserves correctness in vLLM

Given that kept tokens are packed contiguously into the first
`compacted_ctx_len` cache slots, the kernel computes valid attention
over the kept subset. This holds for the following reasons:

**Block table addressing works.** Slot `i` in the compacted cache
contains a valid K/V entry for every `i < compacted_ctx_len`. The block
table entries for those slots point to the correct physical blocks.
There are no gaps to skip over.

**No causal mask in the context phase.** Every query token can attend
to every kept context token. The order of kept tokens in the cache does
not affect correctness — attention is permutation-invariant over the key
set. The dot products and softmax normalization produce the same result
regardless of how the kept tokens are arranged in the cache.

**RoPE is baked into the key vectors.** Each key carries its original
positional encoding from when it was first computed. The relative
position information is preserved in the dot product `Q · K^T`, even
though the key now sits at a different cache slot than its original
position.

**Online softmax normalizes over whatever it sees.** Fewer entries
means a different softmax distribution, but that is the intended effect
of compression. The normalization `acc / (l_i + eps)` is correct for
any subset of attended tokens.

So assuming the compaction is done correctly (kept tokens contiguous,
block table consistent), the kernel computes valid attention over the
kept subset. The only change needed is the loop bound — everything else
in the attention computation remains the same.

## Summary

The difference comes down to cache representation:

- **kvpress** (HuggingFace): the cache is a regular tensor. Compression
  physically shrinks it via `gather`. The tensor shape tells attention
  how many entries there are. No separate length is needed.

- **vLLM**: the cache is a paged structure with fixed-size blocks.
  Compression does not change the structure — it only changes how many
  slots are valid. A separate compacted length tells the kernel where
  to stop iterating.

Both approaches are valid for their respective frameworks. kvpress
benefits from the simplicity of tensor operations. vLLM's approach
preserves the paged cache's memory management advantages (block sharing,
copy-on-write, efficient allocation) at the cost of needing explicit
compacted lengths.
