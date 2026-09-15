# Why `token_presence_bitmap` seems to be the right data structure for key-diff

This note discusses why the `token_presence_bitmap` tensor appears to be a
good fit for a key-diff compression method with a fixed compression ratio.

## What key-diff does

Key-diff is a KV cache compression method that decides which tokens to
keep based on how different their keys are from nearby keys. If a token's
key is similar to its neighbors, the token may be redundant and could
potentially be dropped without significant loss in attention quality.

This decision is made **per head**. A token that appears redundant in one
KV head may be important in another, because different heads tend to attend
to different features of the input.

## What `token_presence_bitmap` represents

The bitmap is a 3D tensor with shape `[num_seqs, num_kv_heads, max_ctx_len]`.
The three dimensions are indices:

- first dimension: which sequence in the batch
- second dimension: which KV head
- third dimension: which token position in the context

Each value is `1` (this token is kept in the cache for this head in this
sequence) or `0` (this token was dropped).

For example, `token_presence_bitmap[2, 5, 10] = 1` means: for sequence 2
in the batch, KV head 5, the token at context position 10 is kept.

Example with 2 KV heads and 6 context tokens:

```text
head 0: [1, 0, 1, 1, 0, 1]   → keeps tokens 0, 2, 3, 5
head 1: [1, 1, 0, 0, 1, 1]   → keeps tokens 0, 1, 4, 5
```

The bitmap captures two things:

- **which** tokens are kept (per head)
- **how many** tokens are kept (derived by counting)

## Two properties of the bitmap

### 1. It supports different sets of kept tokens per head

Different heads can keep different tokens. In the example above, head 0
keeps tokens {0, 2, 3, 5} while head 1 keeps tokens {0, 1, 4, 5}. The
bitmap represents both sets naturally.

This seems important for key-diff, because key similarity is a per-head
property. Token 2 might have a key nearly identical to token 3 in head 0
(so one of them could be dropped), while in head 1 the same two tokens
might have very different keys (so both should probably be kept).

### 2. It requires the same count of kept tokens across heads

The kernel implementation uses a single scalar loop bound per sequence
for the context phase. It cannot iterate 4 times for head 0 and 3 times
for head 1. So `_build_compacted_context_lens` validates that all heads
within a sequence keep the same count, and raises an error otherwise.

This is a constraint: different sets are allowed, but different counts
are not.

## Why the constraint is fine for key-diff with a fixed ratio

A fixed compression ratio means: for each head, keep exactly
`ratio * context_len` tokens. For example, with a 50% ratio and 6 context
tokens, every head keeps exactly 3 tokens.

The selection criteria differ per head (because key similarities differ),
so different heads keep different sets of 3 tokens. But the count is
always 3 for every head, by construction.

Example with 50% compression (keep 3 out of 6):

```text
head 0: [1, 0, 1, 1, 0, 0]   → keeps 3 tokens (0, 2, 3)
head 1: [0, 1, 0, 1, 0, 1]   → keeps 3 tokens (1, 3, 5)
head 2: [1, 1, 0, 0, 0, 1]   → keeps 3 tokens (0, 1, 5)
```

All heads keep 3 tokens. The sets are different. The bitmap captures this
exactly, and the equal-count validation passes.

This appears to be a structural property of fixed-ratio compression. Any
method that keeps a fixed fraction of tokens per head should satisfy the
equal-count constraint, regardless of which tokens each head selects.

## When the constraint might not hold

Compression methods that use a per-head threshold (rather than a fixed
ratio) could produce different counts per head. This does not appear to
be a concern for key-diff with a fixed ratio.

## What the kernel uses from the bitmap

The bitmap does encode the original positions of kept tokens — they can be
reconstructed by looking at where the `1`s are. For example:

```text
bitmap: [1, 0, 1, 1, 0, 1]
→ kept positions: 0, 2, 3, 5
```

However, the bitmap itself is not passed to the kernel. The only thing
extracted from it in the current implementation is the **count** of kept
tokens (4 in this example). The kernel then reads cache slots 0, 1, 2, 3
sequentially, with no knowledge that they originally came from positions
0, 2, 3, 5.

This appears to be sufficient for the common case (no ALiBi, no sliding
window), where the attention computation does not depend on the original
token positions.

If original positions were needed (e.g., for ALiBi), the information is
already present in the bitmap — it would just need to be extracted into
an additional tensor and passed to the kernel. The bitmap has the
information; it is simply not exploited in the current implementation.

## Summary

The `token_presence_bitmap` appears to be well-matched to key-diff with a
fixed compression ratio because:

- it can represent per-head keep/drop decisions (different sets per head)
- a fixed ratio should guarantee equal counts across heads
- the equal-count property is what the kernel requires

The bitmap is a relatively minimal representation: one bit per (sequence,
head, token) position. It seems to store just enough information to derive
the compacted context length, which is the only thing the kernel needs to
adjust its loop bound.
