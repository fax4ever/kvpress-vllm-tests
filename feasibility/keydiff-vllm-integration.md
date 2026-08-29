# Integrating KeyDiff KV Cache Compression into vLLM

## Goal

This document evaluates whether KeyDiff-style KV cache compression can
be integrated into vLLM, and proposes a path to get there.

The short answer: **yes, it's feasible.** The core compression primitive
(gather, score, select, scatter) works — we have validated it
experimentally. The hard part is not the compression itself but
integrating it with vLLM's memory management, scheduling, and metadata
tracking.

## Background

**KeyDiff** is a KV cache compression algorithm that scores tokens by
how different their keys are from their neighbors. Tokens with redundant
keys can be dropped without significant loss in attention quality. The
scoring is per-head: different KV heads may keep different subsets of
tokens, because different heads attend to different aspects of the input.
With a fixed compression ratio, every head keeps the same *count* of
tokens but potentially different *sets* — this property matters for the
integration.

**kvpress** (NVIDIA) implements KeyDiff on top of Hugging Face
Transformers. After each attention layer during prefill, a forward hook
extracts the K/V tensors, scores them, and keeps the top-scoring subset
via `torch.gather`. The tensors physically shrink — dropped tokens are
gone from memory. This is simple and effective, but it only works with
Transformers' eager attention path. It does not integrate with production
serving engines.

**vLLM** is a high-throughput serving engine built around PagedAttention.
The KV cache is a pool of fixed-size physical blocks, managed through
block tables that map logical block indices to physical block IDs.
Sequences do not own contiguous memory — their cache entries are
scattered across blocks allocated from a global pool. This paging is what
makes vLLM efficient (no memory fragmentation, copy-on-write, dynamic
allocation), but it also means you cannot simply "shrink a tensor" to
compress the cache.

## The Gap

kvpress and vLLM solve different problems and are structurally
incompatible:

- In kvpress, compression is a tensor operation:
  `keys = keys.gather(...)`. The tensor shape is the truth — after
  compression, the tensor has fewer entries, and attention operates on
  whatever it receives.

- In vLLM, the cache is a paged structure with fixed-size blocks. Its
  "size" is metadata (`seq_lens`), not a tensor shape. You cannot gather
  from non-contiguous pages with a simple index, and you cannot shrink a
  block table by tensor operations.

At a deeper level, KeyDiff fundamentally breaks vLLM's core invariant:
**cached tokens are immutable once written.** vLLM's architecture —
append-only block tables, prefix caching, reference counting,
scheduling — is built on the assumption that the cache only grows.
Compaction reverses this.

Bridging this gap requires an explicit gather-compress-scatter cycle:
read from the paged cache into a dense buffer, run compression, and
write the survivors back.

## vLLM's Data Structures

Understanding the scatter/gather approach requires knowing the two
representations that vLLM operates on and the metadata that connects
them.

### Dense and Paged Representations

The KV cache lives in two forms. New tokens enter as dense tensors and
are scattered into the paged cache for storage. Attention then reads
from the paged cache:

```text
                 scatter (reshape_and_cache)
    ┌────────┐ ─────────────────────────────▶ ┌────────┐
    │ DENSE  │                                │ PAGED  │──▶ compute attention
    │  K, V  │ ◀───────────────────────────── │ CACHE  │
    └────────┘    gather (PyTorch indexing)    └────────┘
```

The gather direction (paged → dense) does not exist in standard vLLM —
there is no reason to read cache contents back into dense form during
normal inference. It is what we add for compression: read the paged
entries into a dense buffer so that KeyDiff scoring can operate on
regular tensors.

**Dense tensors** — `key, value: [num_tokens, num_kv_heads, head_size]`

These contain only the new tokens being processed in the current forward
pass. `num_tokens` is the total count across all sequences in the batch,
packed contiguously with no padding. `num_kv_heads` is the number of KV
heads (in GQA, `kv_head = query_head // num_queries_per_kv`).
`head_size` is the dimensionality of each individual key, value, or
query vector.

**Paged cache** — `kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]`

The leading dimension stores keys (index 0) and values (index 1)
together. `kv_cache.unbind(0)` splits this into two views, each with shape
`[num_blocks, block_size, num_kv_heads, head_size]`. `num_blocks` is
the total number of physical blocks in the pool (shared across all
sequences). `block_size` is the number of token slots per block.

**Slot mapping** — `slot_mapping: [num_tokens]`

The bridge between dense and paged. Each entry maps one new token to
its physical location in the cache:

```text
slot = block_table[seq_idx, position // block_size] * block_size
       + position % block_size
         └──────── physical block ID ────────┘       └─ offset ─┘
```

The slot mapping makes the cache update sequence-agnostic:
`reshape_and_cache` simply writes each dense token to its designated
slot, without needing to know which sequence it belongs to.

### Cache Update vs Attention Computation

This is an important asymmetry in vLLM's design:

- **Cache update does not distinguish between sequences.** The slot
  mapping routes each token to the right physical location. No
  per-sequence metadata is needed — tokens from different sequences
  are interleaved in the dense tensors and the slot mapping sorts
  them out.

- **Attention must recover per-sequence boundaries.** The kernel
  computes attention separately for each sequence, using per-sequence
  metadata to know where each sequence's tokens start and end, how
  many are cached vs new, and which physical blocks belong to which
  sequence.

The per-sequence metadata that attention uses:

| Tensor | Shape | Purpose |
|--------|-------|---------|
| `num_scheduled_tokens` | `[num_seq]` | How many new tokens each request contributes to this forward pass |
| `query_start_loc` | `[num_seq + 1]` | Cumulative offsets into the packed token dimension `[0, ..., num_tokens]` — marks where each sequence's new tokens begin and end |
| `seq_lens` | `[num_seq]` | Total sequence length per sequence (context + new tokens) |
| `block_table` | `[num_seq, max_blocks_per_seq]` | Row = sequence in the batch, column = physical block ID |

For each sequence, `query_len` is derived from two consecutive entries
in `query_start_loc`. The context length (previously cached tokens) is
`seq_len - query_len`. This distinction between context and query drives
the two-phase structure of prefill attention: context phase (attend to
cached K/V, no causal mask) followed by request phase (attend to new
K/V, causal mask applied).

## What We Know Works

The scatter/gather primitive has been validated experimentally
(`vllm-experiments/01_scatter_gather.ipynb`). We tested:

1. **Round-trip correctness.** Data scattered into the paged cache can be
   gathered back identically, and re-scattered without changing the
   cache. This holds across multiple dtypes (float16, bfloat16), head
   sizes (64, 128), block sizes (16, 32), and head counts (4, 8).

2. **Per-head compaction.** Each KV head can keep a different subset of
   tokens (simulating KeyDiff scoring). After compaction, the first
   `compacted_len` slots contain the correct per-head selections.
   Verified at 25% and 50% compression ratios across all parameter
   combinations (48 test cases).

These experiments used the classic paged attention cache layout (5D keys,
4D values). The same approach needs adaptation for Flash Attention 2's
layout (see below).

## Why Compaction Preserves Correctness

Several properties make this work:

**No causal mask among cached entries.** During both prefill context and
decode, the query attends to all cached tokens without a causal mask
between them. Removing some entries and keeping the rest produces a
valid, smaller context. Attention is permutation-invariant over the key
set — the order of kept tokens does not affect the dot products or
softmax normalization.

**RoPE is baked into the keys.** Each key carries its original positional
encoding from when it was first computed. The relative position
information is preserved in the dot product `Q · K^T`, even though the
key now sits at a different cache slot than its original position.

**Online softmax normalizes over whatever it sees.** Fewer entries means
a different softmax distribution, but that is the intended effect of
compression. The normalization is correct for any subset of attended
tokens.

**Fixed compression ratio guarantees equal counts across heads.** With a
fixed ratio, every head keeps `ratio * seq_len` tokens. The specific
tokens differ per head, but the count is always the same. This matters
because the attention kernel iterates with a single scalar bound per
sequence — it cannot iterate different counts for different heads.

## Proposed Approach

### The Pipeline

For each sequence that needs compression:

1. **Gather keys** from the paged cache into a dense
   `[seq_len, num_kv_heads, head_size]` tensor
2. **Score** the keys using KeyDiff's key-similarity metric
3. **Select** the top-scoring tokens per head →
   `kept_indices: [num_kv_heads, compacted_len]`
4. **Gather values** and apply the same selection
5. **Scatter** the compacted K/V back into slots 0 through
   `compacted_len - 1`
6. **Update metadata** — physical cache occupancy, block table, free
   unused blocks

Keys and values are gathered separately so only one full dense buffer is
in memory at a time. Peak temporary memory per sequence is
`seq_len * num_kv_heads * head_size * dtype_size` (one buffer, reused).

### Where It Fits

The current `Attention.forward()` call tree on the FLASH_ATTN backend is
(the KV cache update is a separate custom op — the backend sets
`forward_includes_kv_cache_update = False`):

```text
Attention.forward()
  ├── torch.ops.vllm.unified_kv_cache_update(key, value, layer_name)
  │     └── FlashAttentionImpl.do_kv_cache_update(key, value, kv_cache, slot_mapping)
  │           ├── key_cache, value_cache = kv_cache.unbind(0)
  │           └── reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)
  └── torch.ops.vllm.unified_attention_with_output(...)
        └── FlashAttentionImpl.forward()
              └── flash_attn_varlen_func(q, key_cache, value_cache, block_table, seqused_k, ...)
```

Compaction inserts between these two operations:

```text
Attention.forward()
  ├── torch.ops.vllm.unified_kv_cache_update(key, value, layer_name)
  │     └── FlashAttentionImpl.do_kv_cache_update(key, value, kv_cache, slot_mapping)
  │           ├── key_cache, value_cache = kv_cache.unbind(0)
  │           └── reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)
  ├── compact_kv_cache(key_cache, value_cache, ...)              ← NEW
  └── torch.ops.vllm.unified_attention_with_output(...)
        └── FlashAttentionImpl.forward()
              └── flash_attn_varlen_func(q, key_cache, value_cache, block_table, seqused_k, ...)
```

This placement works because:

- After `do_kv_cache_update`, the cache contains all tokens (old + new).
  The slot mapping has already routed each token to its physical
  location.
- KeyDiff scoring only needs keys, not attention scores, so compaction
  can run before attention.
- `flash_attn_varlen_func` then computes attention over the
  compacted cache without knowing anything changed — it just sees fewer
  entries via the updated `seq_lens` (`seqused_k`).

Note that `do_kv_cache_update` receives only the flat slot mapping (no
block tables, no sequence lengths). Compaction needs per-sequence
information (block table, sequence length) to gather and scatter for
individual sequences. This metadata is available through
`attn_metadata`, which is accessible via
`get_forward_context().attn_metadata` — both the custom ops and
`Attention.forward()` can reach it.

### NVIDIA / Flash Attention 2

On NVIDIA GPUs, vLLM defaults to Flash Attention 2, which differs from
the classic paged attention in two ways:

**Cache layout.** Both keys and values use the same 4D shape:
`[num_blocks, block_size, num_kv_heads, head_size]`. This is simpler
than the classic layout (5D keys, 4D values with different dimension
ordering). The gather function needs adaptation, but the logic is the
same — index by block and offset, read into dense tensors. The scatter
uses `reshape_and_cache_flash` instead of `reshape_and_cache`.

**Attention kernel.** Flash Attention is a compiled CUDA kernel
(`flash_attn_varlen_func` from the `flash-attn` package), not a Triton
kernel. We cannot modify it. But this is actually *simpler*:
`flash_attn_varlen_func` takes cumulative sequence lengths as parameters
to determine how many KV entries each sequence has. If we compact the
cache and update these values, Flash Attention naturally iterates over
fewer entries. No kernel modification needed.

This is a significant advantage over what would be required on the Triton
path, where the kernels would need explicit code changes to accept a
separate compacted length parameter.

## The Metadata Problem

The hardest part of this integration is not the compression itself — it
is keeping vLLM's metadata consistent after compaction.

vLLM tracks a single sequence length per request. This length serves
multiple purposes:

- **RoPE positions:** the next token's position encoding depends on how
  many tokens have been processed
- **Slot mapping:** determines where the next token's K/V gets written
- **Attention bounds:** how many KV entries the kernel iterates over
- **Block allocation:** how many physical blocks the sequence needs

After compaction, these purposes conflict. If a sequence has processed
100 tokens and compacts to 50:

- RoPE for token 101 must use position 100 (the logical position), not
  50. The model was trained with sequential positions.
- The next token's K/V should be written to slot 50 (right after the
  compacted data), not slot 100 (which would leave a gap).
- Attention should iterate over 51 entries (50 compacted + 1 new token).
- The block allocator can free blocks that held slots 50–99.

This requires separating two concepts that vLLM currently treats as one:

- **Logical sequence length** — total tokens ever processed. Never
  decreases. Used for RoPE positions of new tokens.
- **Physical cache occupancy** — number of valid cache entries. Decreases
  after compaction. Used for attention bounds, slot mapping, and block
  allocation.

The scheduler, the attention metadata builder, and the block allocator
all need to understand this distinction. This is where the bulk of the
implementation effort lives.

## Open Problems

### Block Management

After compaction, physical blocks beyond the compacted range become
unused and should be returned to the block pool. This requires:

- Identifying which blocks are no longer needed
- Updating the block table (truncate unused entries)
- Freeing blocks via the block allocator
- Copy-on-write for shared blocks: prefix caching (enabled by default
  in vLLM) allows multiple sequences to share physical blocks when
  they have the same prefix tokens — the sharing is safe because the
  data is identical. Compaction breaks this invariant by overwriting
  block content with compressed data. Before compacting a block with
  `ref_cnt > 1`, we must fork it: allocate a fresh block, copy the
  data, and update the block table to point to the private copy. Only
  then is it safe to write compacted data.

### When to Compress

Compressing every forward pass would be too expensive — the
gather/scatter overhead would dominate. Reasonable strategies:

- Compress once after prefill (simplest, mirrors kvpress behavior)
- Compress periodically during decode (every N tokens, or when cache
  occupancy exceeds a threshold)
- Compress only when the block allocator is under memory pressure

The right strategy depends on the workload and is an open question.

### Continuous Batching

vLLM processes multiple sequences in each forward pass. Compaction adds
per-sequence overhead, and different sequences may need compression at
different times. The scheduler needs to decide which sequences to
compress and when, without stalling the batch.

### Per-Head Compaction and Cache Coherence

After per-head compaction, each cache slot contains a "composite" entry:
head 0's K/V may come from one original token, head 1's from another.
This is valid because attention processes heads independently.

But it means cache entries no longer correspond to coherent token
positions. Any feature that assumes per-slot coherence (prefix caching
with shared blocks, speculative decoding with cache sharing) would need
careful handling.

### Compression Ratio vs Quality

Our NIAH benchmarks show that 50% compression with KeyDiff preserves
retrieval quality on a needle-in-a-haystack task. But NIAH is a specific
evaluation — the quality impact on other tasks (summarization, multi-hop
reasoning, code generation) needs separate study.

## Roadmap

An incremental path from where we are to a working prototype:

### Completed

1. **NIAH benchmark comparison** — kvpress KeyDiffPress vs vLLM baseline
   (`notebooks/`). Validates that KeyDiff compression preserves retrieval
   quality at various ratios.

2. **Scatter/gather validation** — round-trip and compaction correctness
   on the classic paged attention layout
   (`vllm-experiments/01_scatter_gather`). Validates that the
   gather-compress-scatter primitive is lossless.

### Next Steps

3. **Flash Attention cache layout adaptation** — port the gather/scatter
   functions to the `[num_blocks, block_size, num_kv_heads, head_size]`
   layout used on NVIDIA, with `reshape_and_cache_flash` for scatter.

4. **KeyDiff scoring integration** — replace the random token selection
   in the compaction test with actual key-similarity scoring from
   kvpress. Validate that compacted caches produce correct attention
   outputs end-to-end.

5. **Single-sequence forward pass integration** — insert compaction into
   a vLLM fork's attention layer for a single isolated sequence, with
   hardcoded metadata updates.

6. **Metadata separation** — implement the logical vs physical sequence
   length distinction in the scheduler and attention metadata builder.

7. **Block management** — free unused blocks after compaction, update
   block tables, handle edge cases (partial blocks, last-block
   boundaries). Must handle copy-on-write for blocks shared via prefix
   caching (`ref_cnt > 1`): fork shared blocks into private copies
   before compacting. Note that vLLM v1's block table is append-only,
   so truncating freed blocks may require working around this
   constraint.

8. **Multi-sequence and continuous batching** — handle compaction in the
   presence of batched sequences, chunked prefill, and the scheduler's
   allocation decisions. The scheduler must decide which sequences to
   compress and when, without stalling the batch. Sequences sharing
   prefix blocks add a further constraint: compacting one sequence
   must not affect others that share the same prefix.

Steps 3–5 can likely be tackled in parallel with experimentation.
Steps 6–8 require deeper integration with vLLM internals and are where
the real system-level difficulty lives.
