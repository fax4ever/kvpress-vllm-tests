# Fable's Implementation: filtering

* `kv_compressed`: 
  Used only by the filtering implementation to trigger prefill compress once.

> fitering is done when we're on decoding:
  `num_scheduled == 1 and num_computed_before >= prompt_len`

* `kv_filter_lengths`:  
  Used only by the filtering implementation, initialized lazily on the first filtered decode step.

```python
req_state.kv_filter_lengths = torch.full(
      (len(self.kv_caches), self.kv_caches[0].shape[3]),
      num_cached - 1,    # pre-step cached count (before the new token)
      dtype=torch.long,
      device=self.kv_caches[0].device,
  )
```
* self.kv_caches: kv_caches for each layer
  * len(self.kv_caches): num of layers
* self.kv_caches[0]: has shape `[2, num_blocks, block_size, num_kv_heads, head_size]`
  * self.kv_caches[0].shape[3]: num of heads
* num_cached: total (num_computed_before + num_scheduled) physical tokens
  * num_cached

Shape is [num_layers, num_kv_heads]. Every (layer, head) starts at the same value: the compacted length before this step's new token was added. This is because after prefill compaction the cache is fully packed — all heads have the same valid length.

```python
new_shared_len = filtering_step(
    self.kv_caches,
    block_row,
    block_size,
    lengths,        # <-- kv_filter_lengths
    num_cached,     # <-- num_cached_tokens (physical) / num_cols
    logical_total,  # <-- num_computed_before + num_scheduled
    self.compression_ratio,
)
```

The conseguences:

```python
# The trailing column is freed iff no head extended past the
# previous shared length (kvpress: shrink when the last
# column is all padding).
num_discarded = num_cached - new_shared_len
assert 0 <= num_discarded <= 1
```
## Filtering Step

in `KVCompressionManager#filtering_step`:

1. Compute block indeces and slots

```python
num_cols = num_cached_tokens
# phisycal slots for each token, shape [num_cols]
slots = _slots_for_positions(block_row, block_size, num_cols, device)
block_indices = slots // block_size
block_offsets = slots % block_size
```

2. Compute the `n_kept`:

```python
n_kept = int(round(logical_total_tokens * (1.0 - compression_ratio)))
n_kept = min(max(n_kept, 1), num_cols)

```

3. Loop on layers + unbind:

```python
# for each layer we do have a paged phisycal cache: 
# [2, num_blocks, block_size, num_kv_heads, head_size]
for kv_cache in kv_caches:
  # [num_blocks, block_size, num_kv_heads, head_size]       
  key_cache, value_cache = kv_cache.unbind(0)
  # [num_kv_heads]
  layer_lengths = lengths[layer_idx]

  # 1: GATHER **keys** / **values** (paged -> dense)
  # [num_cached_tokens, num_kv_heads, head_size]
  keys = key_cache[block_indices, block_offsets]

  # build valid mask [num_kv_heads, num_cached_tokens]
  cols = torch.arange(num_cols, device=device)
  valid = cols.unsqueeze(0) < layer_lengths.unsqueeze(1) 
  valid[:, -1] = True 

  # 2: SCORE **keys**
  # [num_kv_heads, num_cached_tokens]
  scores = masked_keydiff_scores(keys, valid)

  # 3: Accept/Reject
  # [num_kv_heads]
  threshold = scores.topk(n_kept, dim=-1).values[:, -1]
  # [num_kv_heads]
  accepts = scores[:, -1] >= threshold
```

Now the PaddingTensor algotirhm `accept_last`:

If on a given head the key is accepted:
  1. the value will be replace the first lefover
  2. layer_lengths[current_head]++
Otherwise:
  **nothing** the value simply remains as leftover-padding 
  layer_lengths[current_head] is not touch!

The impressive part is that my FilteringPress design operated on dense contiguous tensors (PaddedTensor), 
while Fable mapped the same operations directly onto the paged cache through slot indirection.
So we don't need to scatter, working directly on paged memory.

```python
# layer_lengths.shape[0] is the number of heads of current layer 
# [num_kv_heads]
head_indices = torch.arange(layer_lengths.shape[0], device=device)

# destination columns
# accept: first leftover to be replaced; reject: last column
dst_cols = torch.where(accepts, layer_lengths, num_cols - 1)

# destination slots
dst = slots[dst_cols]

# source blocks and offsets
src_blk, src_off = block_indices[-1], block_offsets[-1]

# copy source blocks to destination blocks
key_cache[dst // block_size, dst % block_size, head_indices] = 
  key_cache[src_blk, src_off, head_indices]
value_cache[dst // block_size, dst % block_size, head_indices] = 
  value_cache[src_blk, src_off, head_indices]

# update the lenghts only if accept!
lengths[layer_idx] = layer_lengths + accepts.long()  
```

4. Conseguences

```
return int(lengths.max().item()) # <-- new_shared_len
```







  

