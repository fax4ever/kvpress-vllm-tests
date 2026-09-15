# Implications of a compressed KV cache for decode

This note describes what changes in decode when the KV cache is compressed.

It is about decode only. It does not discuss compressed-cache support for
prefill.

It describes the compressed-cache decode changes implemented on the `key-diff`
branch. It is not a description of the current upstream implementation on
`main`.

In the standard case, the cache keeps keys and values for all past tokens.
Conceptually, token presence looks like:

```text
[1, 1, 1, 1, 1]
```

where `1` means that the token is kept in the KV cache.

With a compressed KV cache, only some past tokens are kept. For example:

```text
[1, 1, 0, 1, 0]
```

This means:

- token 0 is kept
- token 1 is kept
- token 2 is dropped
- token 3 is kept
- token 4 is dropped

So the cache no longer stores keys and values for every token in the original
sequence.

## Main implication

In standard decode attention, the query attends to all cached past tokens.

With a compressed KV cache, the query attends only to the kept past tokens.

So compression changes the context seen by attention:

- standard KV cache: attend over all past tokens
- compressed KV cache: attend over a subset of past tokens

This is the central semantic change.

## What remains true

The attention computation itself still has the same form:

```text
Attention(Q, K, V) = softmax(QK^T)V
```

The difference is not in the formula, but in which keys and values are present.

So compression changes:

- the set of cached keys
- the set of cached values
- the effective context length seen by decode

but it does not change the basic attention mechanism.

In decode, sequence length affects how many past K/V entries contribute to the
weighted sum, but it does not change the size of the resulting output vectors.

## Why this matters

The main benefit is efficiency:

- fewer cached K/V entries to store
- fewer cached K/V entries to load
- fewer attention score contributions to compute

So decode attention can become cheaper in both memory and computation.

## Compact traversal of the cache

If the token-presence pattern is:

```text
[1, 1, 0, 1, 0]
```

then the compressed cache contains only the entries corresponding to:

```text
[0, 1, 3]
```

That means decode must traverse the compacted cache entries rather than the
full original context.

This does **not** mean that decode works unchanged.

It means only that, in the common case, decode may not need to recover the
original logical token positions of kept entries.

## What must still change in decode

Even in the common case, compressed-cache decode still needs some adaptation.

At minimum:

- the decode loop must use the compacted cache length rather than the original
  `seq_len`
- the boundary mask must use that compacted length as well
- the block-table traversal must be interpreted in compacted storage order

In other words, decode still needs to know:

- how many cached entries are physically present
- how to iterate over those compacted entries correctly

So the simplification is **not** "no code change".
The simplification is only:

- no need to reconstruct original token positions, in the common case

## What does not need to change in the common case

If we do not support:

- sliding-window masking
- ALiBi

then the raw attention computation can stay the same.

In particular, decode can still:

- load `Q`
- load the kept `K` and `V`
- compute `QK^T`
- run the online softmax
- accumulate `softmax(QK^T)V`

So the core attention formula is unchanged.
What changes is the traversal of the compressed cache.

## Output shape does not shrink

Compression changes the available past context, not the number of tokens the
model can generate.

So if the original prefix contains 5 past tokens, the next decode step still
produces the next token in the sequence. It does not suddenly produce only 3
tokens because only 3 cached entries were kept.

At the kernel level, the output still has the same logical shape:

```text
[num_current_query_tokens, num_query_heads, head_size]
```

In decode, `num_current_query_tokens` is effectively 1 per sequence.

So compression changes:

- which past K/V entries are attended to

but not:

- the shape of the attention output for the current decode step

## Why common decode is simpler

In decode, causality is already enforced by incremental generation:

- one new query token is processed at a time
- the cache contains only past tokens

So unlike prefill, decode does not need explicit causal masking between many
query positions and many KV positions.

That is why the common decode case is simpler.

If sliding-window masking and ALiBi are also excluded, then compressed decode
does not need the original logical position of each kept KV entry.

The kernel still needs to traverse the compacted cache correctly, but it does
not need extra position-recovery machinery.

## Practical interpretation

A compressed KV cache means:

- not every past token contributes a key and a value
- attention is computed only over the kept tokens
- the model sees an approximation of the original full context

This introduces two main questions:

- which tokens can be removed safely?
- how much memory and compute can be saved by keeping only those tokens?

## Summary

Compressing the KV cache does not change the attention formula itself.

It changes the set of past tokens available to attention.

Instead of storing keys and values for all past tokens, the cache stores them
only for a selected subset. The result is a smaller effective context for
decode, with potential savings in memory and computation.

In the common decode case without sliding-window masking or ALiBi, this can be
implemented without recovering the original logical positions of kept KV
entries.

However, decode still needs small but real adaptations:

- traversal over compacted cache length
- boundary masking with compacted length
- block-table indexing consistent with compacted storage

So the precise statement is:

> compressed-cache decode can be simpler than compressed-cache prefill, because
> it may not require original-position recovery in the common case, but it does
> not work unchanged.
