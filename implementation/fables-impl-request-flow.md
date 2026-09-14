# Fable's Implementation: request flow

At the very end of `KVCompressionManager#run_post_forward` we update the `CachedRequestState#num_kv_discarded` on: 

```python
if num_discarded > 0:
    req_state.num_kv_discarded += num_discarded
    discarded[req_id] = num_discarded
```

Both full replacement and filtering update `num_kv_discarded`!

## Before the step is executed

On `GPUModelRunner#_prepare_inputs`:
(Run for each step — once per batch, not once per request.)

```python
# The kv_discarded_np array has one entry per request in the batch
kv_discarded_np = np.array(
    [self.requests[req_id].num_kv_discarded for req_id in self.input_batch.req_ids],
    dtype=np.int64,
)
if not kv_discarded_np.any():
    kv_discarded_np = None
```

`kv_discarded_np` is used to compute slot mapping on physical tokens:

```python
if kv_discarded_np is not None:
    cache_positions_np = positions_np - kv_discarded_np[req_indices]
    self.input_batch.block_table.compute_slot_mapping(req_indices, cache_positions_np)
else:
    self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
```

After the discard mask is computed, seq_lens is reduced to physical lengths. This is what FlashAttention sees as seqused_k — how many KV entries to iterate over. After compression, it's shorter. The kernel has no idea compression happened — it just reads fewer entries:

```python
if kv_discarded_np is not None:
    # KV compression: attention must iterate only over the
    # physically cached entries.
    self.seq_lens.np[:num_reqs] -= kv_discarded_np
```