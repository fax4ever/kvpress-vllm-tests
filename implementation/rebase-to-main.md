# Rebase of KV Compression Commit to Current Main

**Date**: 2026-09-13
**Original commit**: `7d58349ef` on branch `fax-0.18` (tag v0.18)
**Rebased commit**: `85fbec04e8` on branch `kvpress` (based on current `main`)
**Method**: `git cherry-pick` with manual conflict resolution

## Context

The original commit "[Core] Implement kv compression algorithms" (co-authored
with Claude Fable 5) was written against vLLM v0.18. Since then, main has
undergone significant changes, particularly in `gpu_model_runner.py` where the
slot mapping computation moved from a numpy-based API to a GPU-based Triton
kernel.

## Conflicts Resolved

### 1. `vllm/config/vllm.py` (1 conflict)

**Cause**: Main added a mamba stochastic rounding validation check in the same
location where the KV compression validation checks were inserted.

**Resolution**: Kept both blocks sequentially -- mamba check first, then KV
compression checks. They are independent validations.

### 2. `vllm/v1/core/sched/scheduler.py` (1 conflict)

**Cause**: Main changed `failed_kv_load_req_ids` from `None` to `set[str]()`
and added `failed_recving` handling. The original commit inserted the KV
compression discarded handling right before the old `failed_kv_load_req_ids =
None`.

**Resolution**: Inserted the KV compression discarded block first, then kept
main's `set[str]()` initialization and `failed_recving` handling.

### 3. `vllm/v1/outputs.py` (1 conflict)

**Cause**: Main added `routed_experts`, `sampling_masks` fields and several
static helper methods to `ModelRunnerOutput`. The original commit added
`kv_compression_discarded` after `cudagraph_stats`.

**Resolution**: Added `kv_compression_discarded` field after `cudagraph_stats`,
then kept all of main's additions (`routed_experts`, `sampling_masks`, static
methods).

### 4. `vllm/v1/worker/gpu_model_runner.py` (5 conflicts)

#### Conflict A -- Initialization (line ~621)

**Cause**: Main added `encoder_cudagraph_manager` where the original commit
added `kv_compression_mgr`.

**Resolution**: Kept both initializations sequentially.

#### Conflict B -- Numpy slot mapping (line ~2100)

**Cause**: The original commit replaced the numpy-based
`compute_slot_mapping(req_indices, positions_np)` call with KV compression
position-adjustment logic. Main removed this call entirely -- slot mapping is
now GPU-based via a Triton kernel at a later point.

**Resolution**: Dropped the original numpy-based KV compression code at this
location. The adapted GPU-based logic is inserted in Conflict D instead.

#### Conflict C -- prev_positions (line ~2146)

**Cause**: Main added `optimistic_seq_lens_cpu` fill and
`_compute_prev_positions` for async speculative decoding. These didn't exist in
v0.18.

**Resolution**: Kept main's code.

#### Conflict D -- GPU computation block (line ~2168) [KEY ADAPTATION]

**Cause**: Main restructured the entire `_prepare_inputs` flow: async spec
decode support, mamba preprocessing, GPU-based `num_computed_tokens`, and
GPU-based `positions`/`seq_lens`/`compute_slot_mapping`. The original commit
had numpy-based KV compression adjustments.

**Resolution**: Kept all of main's code, then **adapted** the KV compression
logic from numpy to GPU tensors. The adaptation:

1. After `self.seq_lens` and `self.positions` are computed on GPU, build a
   `kv_discarded_gpu` tensor from per-request `num_kv_discarded` values.
2. Subtract `kv_discarded_gpu` from `self.seq_lens` (attention must iterate
   only over physically cached entries).
3. Create `adjusted_positions = self.positions - kv_discarded_gpu[req_indices]`
   for slot mapping, without modifying `self.positions` (which is used for
   RoPE and must remain at logical positions).
4. Pass `adjusted_positions` to the GPU-based `compute_slot_mapping(num_reqs,
   query_start_loc_gpu, adjusted_positions)`.

The `discard_request_mask` is already correct on main -- it uses
`optimistic_seq_lens_cpu` (the logical sequence length before any KV
compression adjustment).

```python
# Adapted code (GPU-based, replacing the original numpy-based logic)
kv_discarded_gpu: torch.Tensor | None = None
if self.kv_compression_mgr is not None:
    kv_discarded_np = np.array(
        [self.requests[req_id].num_kv_discarded
         for req_id in self.input_batch.req_ids],
        dtype=np.int64,
    )
    if kv_discarded_np.any():
        kv_discarded_gpu = torch.from_numpy(kv_discarded_np).to(
            device=self.device, non_blocking=True
        )

if kv_discarded_gpu is not None:
    self.seq_lens[:num_reqs] -= kv_discarded_gpu.to(self.seq_lens.dtype)
    adjusted_positions = (
        self.positions[:total_num_scheduled_tokens]
        - kv_discarded_gpu[req_indices_gpu]
    )
    self.input_batch.block_table.compute_slot_mapping(
        num_reqs,
        self.query_start_loc.gpu[: num_reqs + 1],
        adjusted_positions,
    )
else:
    self.input_batch.block_table.compute_slot_mapping(
        num_reqs,
        self.query_start_loc.gpu[: num_reqs + 1],
        self.positions[:total_num_scheduled_tokens],
    )
```

#### Conflict E -- Output constructor (line ~4868)

**Cause**: Main added `routed_experts=None` where the original commit added
`kv_compression_discarded`.

**Resolution**: Kept both keyword arguments.

## Auto-Merged Files (No Conflicts)

These files merged cleanly:

- `vllm/config/cache.py` -- KV compression config fields
- `vllm/engine/arg_utils.py` -- CLI argument registration
- `vllm/v1/core/kv_cache_manager.py` -- minor changes
- `vllm/v1/request.py` -- `num_kv_discarded` field on `Request`
- `vllm/v1/worker/gpu_input_batch.py` -- KV compression state on `CachedRequestState`

## New Files (No Conflicts)

- `vllm/v1/worker/kv_compression.py` -- `KVCompressionManager` implementation
- `tests/v1/core/test_kv_compression_scheduler.py` -- scheduler integration tests
- `tests/v1/worker/test_kv_compression_standalone.py` -- standalone compression tests

## API Changes Between v0.18 and Current Main

| Area | v0.18 | Current Main |
|------|-------|-------------|
| Slot mapping | `compute_slot_mapping(req_indices: np.ndarray, positions: np.ndarray)` | `compute_slot_mapping(num_reqs: int, query_start_loc: Tensor, positions: Tensor)` via Triton kernel |
| seq_lens | `SyncedTensor` with `.np`/`.gpu` | Plain GPU tensor |
| positions | Numpy, copied to GPU separately | Computed directly on GPU |
| failed_kv_load_req_ids | `None` | `set[str]()` with `failed_recving` handling |
| ModelRunnerOutput | Fewer fields | Added `routed_experts`, `sampling_masks`, static helpers |

## Validation Status

- All conflict markers removed and verified
- Cherry-pick completed successfully as commit `85fbec04e8`
- GPU-side tests pending (requires CUDA environment)
