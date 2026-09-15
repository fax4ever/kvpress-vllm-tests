# Bitmap vs compacted length: what the kernel actually needs

This note discusses the relationship between the `token_presence_bitmap`
and the compacted length that the kernel uses, and why the current
implementation keeps the bitmap even though the kernel only needs a count.

## What the kernel uses

The Triton kernel for both prefill and decode receives a single integer
per sequence: the compacted length. This is the only information it needs
to adjust its loop bound and boundary masks.

In prefill, this integer is `cur_batch_cache_len`. In decode, it is
`compacted_seq_len`. Neither kernel receives the bitmap itself.

## What the bitmap provides

The `token_presence_bitmap` with shape `[num_seqs, num_kv_heads, max_ctx_len]`
carries richer information:

- **which** tokens are kept, not just how many
- **per-head** keep/drop decisions
- **original positions** of kept tokens (recoverable from where the `1`s are)

All of this is reduced to a single count per sequence by the host-side
helpers (`_build_compacted_context_lens`, `_build_compacted_seq_lens`).

## Could we simplify the interface?

For the kernel side alone, yes. Since the kernel only needs the count, the
interface could accept a `compacted_ctx_lens: [num_seqs]` tensor directly,
with the caller responsible for computing that count. The bitmap and the
counting helpers could be removed from the kernel-side code entirely.

This would be a valid simplification if the compressed-cache support only
targeted the common case (no sliding window, no ALiBi).

## Why we keep the bitmap

The bitmap preserves information that may be needed in the future:

- **Sliding window** support would require knowing the original position
  of each kept token, to evaluate whether it falls within the attention
  window of a given query position. The bitmap encodes this: the `1` at
  index `j` tells us that the kept token originally came from position `j`.

- **ALiBi** support would require the original positions to compute the
  position-dependent bias term `slope * (key_pos - query_pos)`.

- **The compaction logic** (the other side of the thesis) needs to know
  which tokens to keep. The bitmap is the natural output of that logic
  and the natural input to the kernel side. Removing the bitmap from the
  interface would mean adding a separate step to precompute the count,
  losing the direct connection between compaction decisions and kernel
  parameters.

Both sliding window and ALiBi currently raise `NotImplementedError` when
a bitmap is present. Keeping the bitmap in the interface means that
supporting them in the future would only require extracting position
information from the existing bitmap, rather than introducing a new data
structure.

## Summary

The kernel only needs a count, but the bitmap provides more than a count.
The current implementation keeps the bitmap in the interface as a form of
forward compatibility: it does not cost much today (the counting is a
lightweight host-side operation), and it preserves the information that
would be needed to support sliding window and ALiBi in the future.
