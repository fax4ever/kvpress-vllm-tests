# How is `block_table` produced?

In `FlashAttentionImpl.forward` (`vllm/v1/attention/backends/flash_attn.py:714`),
the attention kernel receives a `block_table` tensor that maps each
sequence's logical blocks to physical blocks in GPU memory.

This document traces the call chain that produces and changes this value.

**Note:** This covers the single-GPU (UniProc) case only. In multi-GPU
deployments, the `SchedulerOutput` is serialized through a shared-memory
ring buffer to reach the worker subprocesses. In the single-GPU case,
everything runs in one process via direct function calls.

## At startup (once, when the vLLM process starts)

The block pool is created once for the lifetime of the process,
based on how much GPU memory is available for KV cache. Individual
sequences pop blocks from and return blocks to this pre-existing
pool — no new pool is created per batch or per sequence.

```
vLLM engine init
│
└→ BlockPool.__init__()                                 # block_pool.py:161
    │
    └→ creates KVCacheBlock(0), KVCacheBlock(1), ..., KVCacheBlock(N-1)
        all placed in a free queue
```

## At each inference step

```
EngineCore.step()                                       # engine/core.py:389
│
├→ scheduler.schedule()                                 # scheduler.py:338
│   │
│   └→ _schedule()                                      # scheduler.py:451
│       │
│       └→ for each request (sequence):
│       │   │
│       │   │  [NEW REQUEST] num_computed_tokens = cached  # scheduler.py:804
│       │   │  (0 if no cache hit, or jumps ahead)
│       │   │
│       │   │  [PREEMPTED] num_computed_tokens = 0         # scheduler.py:941
│       │   │  (if request was preempted — KV cache freed)
│       │   │
│       │   │  1) allocate blocks:
│       │   │
│       │   ├→ KVCacheManager.allocate_slots()          # kv_cache_manager.py:218
│       │   │   │
│       │   │   │  READS num_computed_tokens to compute
│       │   │   │  how many blocks are needed.
│       │   │   │
│       │   │   └→ BlockPool.get_new_blocks()           # block_pool.py:320
│       │   │       (only if the last block is full —
│       │   │        pops blocks from the free queue)
│       │   │
│       │   │  2) then advance the cursor:
│       │   │
│       │   └→ num_computed_tokens += num_scheduled       # scheduler.py:964
│       │       (after allocate_slots returns)
│       │
│       └→ returns SchedulerOutput with block IDs
│
└→ executor.execute_model(scheduler_output)
    │
    └→ WorkerWrapperBase.execute_model()                # worker_base.py:327
        │
        └→ Worker.execute_model()                       # gpu_worker.py:822
            │
            └→ GPUModelRunner.execute_model()           # gpu_model_runner.py:3537
                │
                ├→ _update_states(scheduler_output)     # gpu_model_runner.py:3583
                │   │
                │   ├→ block_table.add_row(block_ids)   # block_table.py:82
                │   │   (new requests — writes a full row)
                │   │
                │   └→ block_table.append_row(...)      # block_table.py:100
                │       (existing requests that need more blocks)
                │
                ├→ commit_block_table()                 # block_table.py:193
                │   │
                │   └→ gpu_tensor.copy_(cpu_tensor, non_blocking=True)
                │       Last step that changes the data.
                │
                └→ forward pass
                    │
                    └→ for each attention layer:
                        │
                        └→ Attention.forward()
                            │
                            ├→ do_kv_cache_update(key, value, kv_cache, slot_mapping)
                            │   │
                            │   ├→ split_kv_cache(kv_cache) → key_cache, value_cache
                            │   │
                            │   └→ reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
                            │
                            └→ chunked_prefill_paged_decode(query, key, value, ...)
                                (reads block_table to locate K/V blocks)
```

Block allocation is **conditional** — during decoding (1 token per step),
the last block usually has room and no new allocation happens.

If there aren't enough free blocks, the scheduler either skips the
request this step or preempts a lower-priority sequence to free blocks.

## How `allocate_slots` decides whether to allocate

`KVCacheManager.allocate_slots` (`kv_cache_manager.py:218`) runs once
per sequence per inference step. It is the single point that decides
whether a sequence needs more physical memory.

The core algorithm (for a running sequence, the common case) is simple
arithmetic in `SingleTypeKVCacheManager.get_num_blocks_to_allocate`
(`single_type_kv_cache_manager.py:105`):

```python
num_required_blocks = ceil(total_tokens / block_size)
num_existing_blocks = len(blocks_already_allocated)
blocks_to_allocate  = max(num_required_blocks - num_existing_blocks, 0)
```

The decision is based on three values from three different sources:

- **total_tokens**: `request.num_computed_tokens` (from the **request
  object**, updated after each step) + `num_new_tokens` (from the
  **scheduler** — 1 during decoding, prompt length during prefill)
- **num_existing_blocks**: from the **cache manager**'s internal
  `req_to_blocks` dict — updated each time blocks are allocated
  or freed for this sequence
- **block_size**: configuration constant (e.g. 16)

If the existing blocks have room, `blocks_to_allocate` is 0 and
nothing happens. A new block is only allocated when the last block
is full — i.e., when `ceil(total_tokens / block_size)` exceeds the
current block count.

During decoding this means: with `block_size=16`, a new block is
allocated every 16 steps. The other 15 steps require zero allocation.

If `blocks_to_allocate > 0` but there aren't enough free blocks in
the pool, `allocate_slots` returns `None` and the scheduler handles
it — either skipping the request or preempting another sequence.

## `request.num_computed_tokens`

Defined on the `Request` class (`vllm/v1/request.py:135`), initialized
to 0. It is a cursor that tracks how many tokens of the sequence have
valid KV cache entries.

There is no "decoding phase" or "prefill phase" in the scheduler — each
request just has `num_computed_tokens`. The scheduler's job is to close
the gap between this value and the total tokens the sequence needs.
This one counter unifies prefill, decoding, chunked prefill, prefix
caching, and preemption.

### Where it is updated

**Set to cached tokens when a request is first scheduled:**

```python
request.num_computed_tokens = num_computed_tokens       # scheduler.py:804
```

0 if no prefix cache hit, or jumps ahead if cached tokens are found.

**Incremented after scheduling by the tokens to be computed this step:**

```python
request.num_computed_tokens += num_scheduled_token      # scheduler.py:964
```

This is the normal growth — 1 during decoding, prompt length (or chunk)
during prefill.

**Reset to 0 on preemption (KV cache freed, must recompute):**

```python
request.num_computed_tokens = 0                         # scheduler.py:941
```

**Decremented on speculative decode rejection:**

```python
request.num_computed_tokens -= num_rejected             # scheduler.py:1351
```

Optimistically advanced tokens are rolled back when rejected by the
verifier.

## How the block table evolves over a sequence's lifetime

The scheduler does not know the total length of a sequence in advance.
Sequences grow incrementally (prefill, then decoding token by token,
possibly continuation prefills, more decoding, until end of sentence).
The block table grows accordingly:

```
Step 1 (prefill 37 tokens, block_size=16):
  allocate 3 blocks -> block_table = [7, 2, 15]
  block 15: 11 empty slots

Step 2-12 (decode 1 token each):
  block 15 still has room -> no allocation
  block_table = [7, 2, 15]  (unchanged for 11 steps)

Step 13 (decode 1 token):
  block 15 is full -> allocate 1 new block
  block_table = [7, 2, 15, 9]

... sequence continues growing ...

End of sequence:
  blocks [7, 2, 15, 9, ...] returned to free pool
```
