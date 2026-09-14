# Fable's Implementation: full replacement

First of all, we introduce a state in `CachedRequestState`:

Each request carries four compression fields, all defaulting to "nothing happened yet":
- num_kv_discarded: int = 0 — total entries discarded so far
- kv_compressed: bool = False — whether compaction has run
- kv_last_compaction_total: int = 0 — logical token count at last compaction
- kv_filter_lengths — only used by filtering, valid lengths [num_layers, num_kv_heads]

Fable's key idea was to compress after `GPUModelRunner#_model_forward`.
--> `KVCompressionManager#run_post_forward` that returns the `_kv_compression_discarded` update.

in `KVCompressionManager#run_post_forward`:

```python
# FOR EACH REQUEST
for req_index, req_id in enumerate(input_batch.req_ids): 

    # Context logical tokens
    num_computed_before = int(input_batch.num_computed_tokens_cpu[req_index])

    # Request logical tokens
    num_scheduled = scheduler_output.num_scheduled_tokens[req_id]

    # Total logical tokens
    logical_total = num_computed_before + num_scheduled

    # Total physical tokens
    num_cached = logical_total - req_state.num_kv_discarded

    # Did prefill complete in this exact step ?
    prefill_completes = num_computed_before < prompt_len <= logical_total

    # Should I compress at this step ?
    compaction_due = (logical_total - req_state.kv_last_compaction_total >= self.compression_interval)
```

## In case of `full_replacement`:

```python
# Physical block ID assigned to that request's i-th logical block
# shape: [max_blocks_per_request]
block_row = block_table_np[req_index]

# compact is triggered at the round of prefill completes or for compression_interval
if compaction_due or prefill_completes:
    num_discarded = self._compact(
        req_state, block_row, block_size, num_cached, logical_total
    )
```

in `KVCompressionManager#_compact`:

```python
# physical tokens = (1 - compression_ratio) * logical tokens
n_kept_target = max(1, int(logical_total * (1.0 - self.compression_ratio)))
n_kept = compact_request_kv(
        self.kv_caches,
        block_row,            
        block_size,
        num_cached, # <-- num_cached_tokens
        n_kept_target,
    )
```

in `KVCompressionManager#compact_request_kv`:

```python
# physical slots for each token, shape [num_cached_tokens], before compression
src_slots = _slots_for_positions(block_row, block_size, num_cached_tokens, device)
# physical slots for each token, shape [n_kept], after compression
dst_slots = src_slots[:n_kept]

# for each layer we have a paged physical cache: 
# [2, num_blocks, block_size, num_kv_heads, head_size]
for kv_cache in kv_caches:
    # [num_blocks, block_size, num_kv_heads, head_size]       
    key_cache, value_cache = kv_cache.unbind(0)

    # 1: GATHER **keys** / **values** (paged -> dense)
    # [num_cached_tokens, num_kv_heads, head_size]
    keys = gather_slots(key_cache, src_slots)
    values = gather_slots(value_cache, src_slots)

    # 2: SCORE **keys**
    # [num_kv_heads, num_cached_tokens]
    scores = keydiff_scores(keys)

    # 3: SELECT (sorted to preserve temporal order) **keys** / **values**
    # indices of survivors [num_kv_heads, n_kept]
    kept_idx = scores.topk(n_kept, dim=-1).indices.sort(dim=-1).values
    kept_keys = keys.transpose(0, 1).gather(1, kept_idx).transpose(0, 1)
    kept_values = values.transpose(0, 1).gather(1, kept_idx).transpose(0, 1)

    # 4: SCATTER **keys** / **values** (dense -> paged)
    scatter_slots(key_cache, dst_slots, kept_keys.contiguous())
    scatter_slots(value_cache, dst_slots, kept_values.contiguous())

return n_kept    
```

The result is pushed back to the request object:

Step 5 — State update (kv_compression.py:439-442)

num_discarded = num_cached - n_kept
req_state.num_kv_discarded += num_discarded
req_state.kv_last_compaction_total = logical_total  # reset the interval trigger

Step 6 — Report to scheduler (kv_compression.py:442)

The discarded dict ({req_id: num_newly_discarded}) is returned, stored as self._kv_compression_discarded, and later attached to ModelRunnerOutput so the scheduler can update its copy for block allocation.




