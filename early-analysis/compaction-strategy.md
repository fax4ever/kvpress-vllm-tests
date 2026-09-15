# A compaction strategy for key-diff in vLLM

This note explores a possible approach for compacting the paged KV cache
after key-diff eviction, using existing vLLM primitives. It is a design
sketch — further investigation and testing would be needed to validate
the assumptions made here.

## The problem

After key-diff scoring decides which tokens to keep and which to drop,
the kept tokens need to be packed contiguously into the first
`compacted_len` slots of the paged KV cache. The attention kernel
iterates from slot 0 to `compacted_len - 1` without skipping gaps.

## Per-head compaction

Key-diff with a fixed compression ratio produces the same *count* of
kept tokens per head, but different *sets* of kept tokens. Different
heads attend to different aspects of the input, so they assign different
importance scores to the same tokens.

Each compacted slot becomes a composite: head 0's data at compacted
slot 1 might come from original token 2, while head 1's data at the
same slot comes from original token 4. This should be valid, since each
head appears to be processed independently in the attention kernel.

## The existing forward pass

The current flow in `Attention.forward()` is:

```text
Attention.forward()
  ├── do_kv_cache_update(key, value, kv_cache, slot_mapping)
  │     ├── split_kv_cache(kv_cache) → key_cache, value_cache
  │     └── reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
  └── chunked_prefill_paged_decode(query, key, value, key_cache, value_cache, ...)
```

### What `do_kv_cache_update` does and does not have

`do_kv_cache_update` receives:

- `key`, `value` — **dense**, shape `[num_tokens, num_kv_heads, head_size]`.
  These contain **only the new tokens** (the "diff") being processed in
  the current forward pass. They are not yet in the cache. They may span
  **multiple sequences** packed contiguously in dim 0.
- `kv_cache` — **paged**. The full paged cache tensor.
- `slot_mapping` — `[num_tokens]`, the bridge from dense to paged. Each
  entry is a flat slot index (`block_idx * block_size + offset`) telling
  the kernel where to write each token.

`do_kv_cache_update` does **not** receive per-sequence metadata
(`query_start_loc`, `seq_lens`, block tables). It does not need them
because `slot_mapping` already routes each token to the correct
per-sequence cache slot.

The operation is **append-only**: it scatters the new tokens into the
cache and never touches existing entries.

### What `chunked_prefill_paged_decode` additionally has

The attention kernel receives per-sequence metadata:

- `query_start_loc` — `[num_seqs + 1]`, cumulative offsets into the
  packed token dimension (maps token index to sequence)
- `seq_lens` — `[num_seqs]`, total sequence length per sequence
- `block_table` — per-sequence logical-to-physical block mapping

It needs these to compute attention separately per sequence with the
right bounds, masks, and block table lookups.

### The dense tensor dimensions

The dense `key` and `value` tensors have shape
`[num_tokens, num_kv_heads, head_size]`:

- **`num_tokens`** (dim 0): total tokens in the current forward pass,
  packed across all sequences with no padding. During prefill with
  sequence A (100 tokens) and B (50 tokens), `num_tokens = 150`. During
  decode, each sequence contributes 1 token, so `num_tokens` equals the
  number of decoding sequences. It is the total token count, not the
  batch size.

- **`num_kv_heads`** (dim 1): the number of KV heads. In GQA, multiple
  query heads share one KV head
  (`kv_head = query_head // num_queries_per_kv`). Key-diff eviction
  decisions are made independently per KV head.

- **`head_size`** (dim 2): the hidden dimension per head (e.g. 128 for
  a model with hidden size 4096 and 32 heads). In the 5D key cache,
  this is split into `head_size // x` groups of `x` elements where
  `x = 16 // element_size`.

### The cache layout

The ROCm KV cache has shape `(2, num_blocks, block_size, num_kv_heads, head_size)`.
After `split_kv_cache`:

- `key_cache`: `[num_blocks, num_kv_heads, head_size//x, block_size, x]`
- `value_cache`: `[num_blocks, num_kv_heads, head_size, block_size]`

### The slot mapping

`slot_mapping` has shape `[num_tokens]` — one integer per new token in
`key`/`value`. Each integer identifies the cell in the paged cache
dedicated to storing that token's key and value data across all KV
heads. The same `block_idx` and `offset` address the corresponding
entries in both the 5D key cache and the 4D value cache.

A slot is decomposed as:

```text
slot = block_idx * block_size + offset
```

where `offset` ranges from `0` to `block_size - 1` (the position of the
token within its physical block). The slot is computed from the
sequence's block table:

```text
slot = block_table[req_idx, position // block_size] * block_size
       + position % block_size
```

## Worked example: batch metadata and block table

Consider a batch of 3 sequences with `block_size = 16`:

- Sequence 0: **fresh prefill** — 4 new tokens, nothing cached yet
- Sequence 1: **chunked prefill** — 10 new tokens, 20 already cached
- Sequence 2: **decode** — 1 new token, 50 already cached

The metadata tensors for this batch are:

```text
num_scheduled_tokens = [4, 10, 1]
query_start_loc      = [0, 4, 14, 15]
seq_lens             = [4, 30, 51]
```

### `query_start_loc`

Shape: `[num_seqs + 1]`. A cumulative-sum of per-sequence query lengths.
It tells the kernel where each sequence's tokens begin and end within the
packed `Q`, `K`, `V` tensors (shape `[15, num_heads, head_size]` here,
with 15 = total tokens and no padding between sequences).

Derived per-sequence values:

| Seq | query_len | seq_len | ctx_len | Description |
|-----|-----------|---------|---------|-------------|
| 0   | `4 - 0` = 4  | 4  | `4 - 4` = 0  | Fresh prefill — no context phase |
| 1   | `14 - 4` = 10 | 30 | `30 - 10` = 20 | Chunked prefill — context phase over 20 cached slots |
| 2   | `15 - 14` = 1 | 51 | `51 - 1` = 50 | Decode — context phase over 50 cached slots |

The kernel loads two consecutive entries to find a sequence's token range:

```python
cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
cur_batch_in_all_stop_index  = tl.load(B_Start_Loc + cur_batch + 1)
cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
cur_batch_ctx_len   = cur_batch_seq_len - cur_batch_query_len
```

### `block_table`

Shape: `[num_seqs, max_num_blocks_per_seq]`. Maps logical blocks to
physical block IDs from the pool. The second dimension is the maximum
across all sequences (4 here, from sequence 2).

Each sequence needs `ceil(seq_len / block_size)` blocks:

- Sequence 0: `ceil(4/16)` = 1 block
- Sequence 1: `ceil(30/16)` = 2 blocks
- Sequence 2: `ceil(51/16)` = 4 blocks

Physical blocks are drawn from a global `BlockPool`. In the general
case, each physical block is dedicated to a single sequence — sequences
do not share blocks. The exception is prefix caching: when enabled,
sequences that share a common prompt prefix can reference the same
physical blocks (tracked via `ref_cnt` on each block). For key-diff
compaction, shared prefix blocks would need special handling (e.g.,
copy-on-write before modifying), but in the common non-prefix-caching
case, each sequence owns its blocks exclusively.

Physical block IDs are arbitrary integers assigned by the allocator:

```text
                      logical block:    0     1     2     3
                                      ┌─────┬─────┬─────┬─────┐
block_table = seq 0:                  │   7 │  -  │  -  │  -  │
              seq 1:                  │   3 │  12 │  -  │  -  │
              seq 2:                  │   0 │   5 │   9 │   2 │
                                      └─────┴─────┴─────┴─────┘
```

Entries marked `-` are unused (never read because `seq_len` bounds
the kernel's iteration).

### How token positions map to cache slots

For any token at position `p` in a sequence:

```text
logical_block  = p // block_size
offset         = p  % block_size
physical_block = block_table[seq_idx, logical_block]
slot           = physical_block * block_size + offset
```

Example slot lookups:

| Seq | Token pos | Logical block | Offset | Physical block | Slot |
|-----|-----------|---------------|--------|----------------|------|
| 0   | 0         | 0             | 0      | 7              | 112  |
| 0   | 3         | 0             | 3      | 7              | 115  |
| 1   | 0         | 0             | 0      | 3              | 48   |
| 1   | 15        | 0             | 15     | 3              | 63   |
| 1   | 16        | 1             | 0      | 12             | 192  |
| 1   | 29        | 1             | 13     | 12             | 205  |
| 2   | 0         | 0             | 0      | 0              | 0    |
| 2   | 32        | 2             | 0      | 9              | 144  |
| 2   | 50        | 3             | 2      | 2              | 34   |

Consecutive token positions can land in non-contiguous physical memory —
that is the purpose of paging.

### How the kernel uses the block table

During the context phase, the kernel tiles over cached token positions
and translates each to a physical block ID:

```python
token_indices = start_n + offs_bs_n
bn_logical_indices = token_indices // PHYSICAL_BLOCK_SIZE
bn = tl.load(B_Loc + cur_batch * stride_b_loc_b
             + bn_logical_indices * stride_b_loc_s).to(tl.int64)
```

For sequence 1 with `start_n = 0` and a tile of 16 tokens:
`token_indices = [0..15]`, `bn_logical_indices = [0..0]` (all in logical
block 0), `bn = [3..3]` (physical block 3). Next tile at `start_n = 16`:
`bn_logical_indices = [1..1]`, `bn = [12..12]` (physical block 12).

### Compaction effect on the block table

After compacting sequence 2 at 50% compression, 25 of 50 context tokens
survive. They are packed into slots 0–24, spanning `ceil(25/16)` = 2
blocks. The last 2 physical blocks (9 and 2) become unused and can be
freed back to the block pool:

```text
before: block_table[2] = [0, 5, 9, 2]   (4 blocks, 50 tokens)
after:  block_table[2] = [0, 5, -, -]   (2 blocks, 25 tokens)
```

## Where compaction fits

The KV cache update and attention computation are called as separate
operations from the attention layer (`Attention.forward` in
`attention.py`):

```text
Attention.forward()
  ├── unified_kv_cache_update(key, value, ...)  — calls do_kv_cache_update
  ├── compact_kv_cache(...)                     — NEW: per-sequence compaction
  └── unified_attention_with_output(...)        — calls chunked_prefill_paged_decode
```

It looks like this placement could work because:

1. After `unified_kv_cache_update`, the cache should contain all tokens
   for each sequence (old + just-written).
2. Key-diff scoring only needs keys — it does not need attention scores.
   So it seems possible to run compaction before attention.
3. `unified_attention_with_output` would then compute attention on the
   compacted cache without knowing anything changed.

Compaction likely cannot live inside `do_kv_cache_update` because that
method lacks per-sequence metadata. It would need to be orchestrated at
a level that has access to per-sequence block tables and sequence
lengths. The attention metadata (`attn_metadata`) is available through
`get_attention_context` and contains both `seq_lens` and `block_table`.

### Computing the target compacted length

With a fixed compression ratio, we need the **original uncompressed
context length** to decide how many tokens to keep:

```python
compacted_len = int(original_ctx_len * (1 - compression_ratio))
```

`attn_metadata.seq_lens` provides the logical (uncompressed) sequence
length per sequence. This value keeps growing as new tokens arrive,
regardless of whether the cache was previously compacted. In contrast,
the physical cache occupancy after compaction may be smaller.

After compaction, the system needs to track two values per sequence:

- **`seq_lens[i]`** — the logical sequence length (all tokens ever
  seen), used for RoPE positions and for computing the compaction
  target. This never decreases.
- **`compacted_len`** — the physical number of valid cache slots, used
  by the kernel as its loop bound. This is the result of applying the
  compression ratio to `seq_lens[i]`.

Without preserving `seq_lens[i]` as the original length, applying the
ratio to an already-compacted length would compound the compression —
each compaction step would shrink the cache further, which is not the
intended behavior.

## The compaction operation

Compaction would operate **per sequence**. Each sequence's cached
context appears to be independent in the paged cache, addressed through
its own block table. A sequence using 10 blocks out of a pool of
thousands should only need to touch those 10 blocks.

The steps are sequenced so that keys and values are gathered separately,
avoiding having both full dense buffers in memory at the same time.
Key-diff scoring only needs keys, so keys are gathered and scored first.
Values are gathered only after the kept indices are known.

### Step 1: Gather keys (paged → dense)

Build a slot mapping for the sequence's cached positions 0 through
`seq_len - 1`, then read keys from the paged cache:

```python
block_indices = slot_mapping // block_size
offsets = slot_mapping % block_size

# Keys (5D layout)
keys_packed = key_cache[block_indices, :, :, offsets, :]
keys = keys_packed.reshape(seq_len, num_kv_heads, head_size)
# → [seq_len, num_kv_heads, head_size]
```

### Step 2: Score and select keys

Key-diff scoring runs on the dense keys and produces
`kept_indices: [num_kv_heads, compacted_len]`. The kept keys are
selected per head using `torch.gather`:

```python
keys_by_head = keys.permute(1, 0, 2)      # [num_kv_heads, seq_len, head_size]
idx = kept_indices.unsqueeze(-1).expand(-1, -1, head_size)
keys_kept = keys_by_head.gather(1, idx)    # [num_kv_heads, compacted_len, head_size]
keys_compact = keys_kept.permute(1, 0, 2).contiguous()
# → [compacted_len, num_kv_heads, head_size]
```

The full keys buffer can be freed after this step.

### Step 3: Gather and select values

Now gather values using the same slot mapping, and apply the same
`kept_indices`:

```python
# Values (4D layout)
values = value_cache[block_indices, :, :, offsets]
# → [seq_len, num_kv_heads, head_size]

values_by_head = values.permute(1, 0, 2)
values_kept = values_by_head.gather(1, idx)
values_compact = values_kept.permute(1, 0, 2).contiguous()
# → [compacted_len, num_kv_heads, head_size]
```

The full values buffer can be freed after this step.

### Step 4: Scatter back (dense → paged)

Write the compacted tokens into slots 0 through `compacted_len - 1`
using the existing `reshape_and_cache`:

```python
compact_slot_mapping = torch.arange(compacted_len, device=device)
reshape_and_cache(keys_compact, values_compact,
                  key_cache, value_cache,
                  compact_slot_mapping, ...)
```

### Step 5: Update metadata

- Set `compacted_ctx_len` for the sequence
- Free physical blocks that are no longer needed
- Update the block table and `req_to_blocks`

## What exists and what needs to be written

### Already exists

- **The scatter**: `reshape_and_cache` (HIP C++ kernel) and
  `triton_reshape_and_cache_flash` (Triton fallback)
- **`split_kv_cache`**: splits the combined cache into key/value views
- **Block table**: maintained by the scheduler and worker
- **Slot mapping for new tokens**: `_compute_slot_mapping_kernel`
- **`token_presence_bitmap`**: on the `key-diff` branch, with compacted
  lengths derived from the bitmap

### Needs to be written

- **A gather function for the ROCm paged cache**: PyTorch advanced
  indexing on the 5D key cache and 4D value cache, as shown in Step 1
- **Slot mapping for cached tokens**: same formula as for new tokens,
  but for positions 0 through `seq_len - 1` using the sequence's
  existing block table
- **The orchestration** (`compact_kv_cache`): ties gather, scoring,
  selection, and scatter together, called from `Attention.forward()`
  between `do_kv_cache_update` and `chunked_prefill_paged_decode`

## Implementation plan

### First step: round-trip validation

The simplest possible thing to build first. No scoring, no selection,
no compaction. Just verify that gather and scatter are compatible:

1. Pick a sequence that has cached tokens
2. Build a slot mapping for its positions 0 through `seq_len - 1`
3. Gather all keys and values into dense tensors
4. Scatter them back with `reshape_and_cache` using the same slot mapping

This should be a no-op — the data goes out and comes back to the same
slots. Verify numerically that the cache contents are unchanged.

If this works, it would suggest that gather and scatter are compatible,
and every subsequent step (scoring, selection, compacted scatter) should
be safe to add incrementally.

### Second step: add key-diff scoring and per-head selection

Once the round-trip is validated, insert key-diff scoring between
gather and scatter. The kept indices from scoring drive `torch.gather`
on the dense keys and values, and the compacted result is scattered
back to slots 0 through `compacted_len - 1`.

### Third step: metadata and block management

Update `compacted_ctx_len`, free unused blocks, update the block table.

## Block management

After compaction, blocks beyond `compacted_len` may become unused:

- If a sequence occupied 10 blocks and compacts to 6, the last 4 can
  be freed back to the `BlockPool`
- Partial blocks are fine — the kernel handles them via boundary masks

## Memory overhead

Because keys and values are gathered and compacted separately, peak
temporary memory is:

- 1 full buffer: `seq_len * num_kv_heads * head_size * sizeof(dtype)`
- 1 compacted buffer: `compacted_len * num_kv_heads * head_size * sizeof(dtype)`

Not both full buffers at the same time. Each full buffer is freed before
the next is allocated.

A custom Triton kernel could potentially eliminate these buffers by
copying directly between cache slots, but the simpler approach seems
sufficient for a first implementation.

## Summary

It appears that compaction could run between `do_kv_cache_update` and
`chunked_prefill_paged_decode` in `Attention.forward()`. Key-diff
scoring only needs keys and does not need attention scores, so it looks
like compaction can happen before attention runs.

The proposed operation per sequence is:

1. **Gather keys** from paged cache (PyTorch indexing)
2. **Score and select** kept keys (key-diff + `torch.gather`)
3. **Gather and select** kept values (same indices)
4. **Scatter back** (`reshape_and_cache`, already exists)
5. **Update metadata** (compacted length, block table)

Keys and values would be gathered separately so only one full dense
buffer is in memory at a time.

This approach needs validation, starting with the round-trip test
described in the implementation plan.
