# Decode attention in `chunked_prefill_paged_decode`

This note focuses on the decode path of
`vllm/v1/attention/ops/chunked_prefill_paged_decode.py`.

The function contains both prefill and decode logic, but the rest of this
document is about the decode case.

For one sequence, the choice is exclusive:

- if `query_len > 1`, use the prefill path
- if `query_len == 1`, use the decode path

At the batch level, a single call may contain both kinds of sequences.

## Prefill / Decode

The function first launches prefill when the batch contains at least one
multi-token query:

```python
if max_query_len > 1:
    context_attention_fwd(..., skip_decode=True)
```

Conceptually, this handles the sequences with `query_len > 1`.

Then it launches paged decode:

```python
kernel_paged_attention_2d(...)
```

Conceptually, this handles the sequences with `query_len == 1`.

## KV cache layout

For the Triton decode path, the cache tensors are interpreted as:

- `key_cache`: `[num_blocks, num_kv_heads, head_size // x, block_size, x]`
- `value_cache`: `[num_blocks, num_kv_heads, head_size, block_size]`

where:

- `num_blocks` = total number of physical cache blocks
- `num_kv_heads` = number of KV heads
- `head_size` = hidden dimension per head
- `block_size` = number of token slots in one physical cache block
- `x` = packed inner dimension for keys

Each physical block stores token slots for all KV heads.
The kernel grid is over `(sequence, kv_head)`, and each program instance
processes one `(seq_idx, kv_head_idx)` pair.

## Triton decode iteration structure

In the Triton kernel, attention is computed over tiles of tokens rather than
directly over physical cache blocks:

```python
num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
for j in range(0, num_blocks):
    ...
```

Important distinction:

- `PHYSICAL_BLOCK_SIZE` belongs to the vLLM KV-cache organization
- `BLOCK_SIZE` belongs to the Triton kernel's compute tiling

These are equal in common layouts, but they can differ for non-standard cases
such as a physical block size of 544.

## Loading K and V

The Triton kernel computes addresses into the packed KV cache as follows:

```python
# 5D addressing logic of K
k_offset = (
    p_block_idx[None, :] * stride_k_cache_0
    + kv_head_idx * stride_k_cache_1
    + (offs_d[:, None] // x) * stride_k_cache_2
    + internal_offsets[None, :] * stride_k_cache_3
    + (offs_d[:, None] % x) * stride_k_cache_4
)

# 4D addressing logic of V
v_offset = (
    p_block_idx[:, None] * stride_v_cache_0
    + kv_head_idx * stride_v_cache_1
    + offs_d[None, :] * stride_v_cache_2
    + internal_offsets[:, None] * stride_v_cache_3
)
```

Broadcasting note for the indexing vectors:

- `p_block_idx` is a vector because the kernel processes a tile of
  `BLOCK_SIZE` token positions at once, and each token position maps to a
  physical KV-cache block
- In this kernel, if `BLOCK_SIZE == PHYSICAL_BLOCK_SIZE`, then
  `p_block_idx` is all equal within one tile
- `p_block_idx`: `[BLOCK_SIZE]`, so `p_block_idx[None, :]` becomes
  `[1, BLOCK_SIZE]` and `p_block_idx[:, None]` becomes `[BLOCK_SIZE, 1]`
- `kv_head_idx` is a scalar index selecting one KV head for the current
  program instance
- `offs_d` enumerates positions in the head dimension for the current tile
- `offs_d`: `[HEAD_SIZE_PADDED]`, so `offs_d[:, None]` becomes
  `[HEAD_SIZE_PADDED, 1]` and `offs_d[None, :]` becomes `[1, HEAD_SIZE_PADDED]`
- `internal_offsets` gives the position of each token inside its physical
  cache block
- `internal_offsets`: `[BLOCK_SIZE]`, so `internal_offsets[None, :]` becomes
  `[1, BLOCK_SIZE]` and `internal_offsets[:, None]` becomes `[BLOCK_SIZE, 1]`
- `x` is the packed inner dimension of the key cache layout; it is used to
  split one head-dimension index into:
  - a packed group index: `offs_d // x`
  - an offset inside that packed group: `offs_d % x`

These singleton dimensions let Triton combine token-index vectors with
head-dimension vectors to build 2D address grids.

Then it loads one tile:

```python
# K : (HEAD_SIZE_PADDED, BLOCK_SIZE)
K_load = tl.load(
    key_cache_ptr + k_offset,
    mask=dim_mask[:, None],
    other=0.0,
    eviction_policy="evict_last",
)

if K_load.dtype.is_fp8():
    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
else:
    K = K_load

# V : (BLOCK_SIZE, HEAD_SIZE_PADDED)
V_load = tl.load(
    value_cache_ptr + v_offset,
    mask=dim_mask[None, :],
    other=0.0,
    eviction_policy="evict_last",
)

if V_load.dtype.is_fp8():
    V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
else:
    V = V_load
```

Logical interpretation for one tile:

- `Q`: `[num_queries_per_kv_padded, head_size]`
- `K`: `[head_size, BLOCK_SIZE]`
- `V`: `[BLOCK_SIZE, head_size]`

Conceptually, over the full sequence for one KV head:

- cached keys are organized by token, i.e. `[seq_len, head_size]`
- the kernel loads a tile and uses a transposed view of K for the dot product
- cached values are conceptually `[seq_len, head_size]`

## Attention computation

For each tile, the kernel computes:

```python
qk = scale * tl.dot(Q, K)
S = tl.where(head_mask[:, None] & seq_mask, qk, float("-inf"))
```

Then optional modifiers are applied:

Sliding-window masking:

```python
S = tl.where(
    (context_len - seq_offset) < SLIDING_WINDOW,
    S,
    -10000,
)
```

ALiBi bias:

```python
S += alibi_slope[:, None] * (seq_offset - context_len)
```

The softmax is computed online across tiles:

```python
# running max
m_j = tl.maximum(M, tl.max(S, axis=1))

# local probabilities
p = tl.exp(S - m_j[:, None])
p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)

# local sum
l_j = tl.sum(p, axis=1)

# rescaling factor for previous accumulator
alpha = tl.exp(M - m_j)
alpha = tl.where(float("-inf") == M, 0.0, alpha)

acc = acc * alpha[:, None]
L = L * alpha + l_j
M = m_j

acc += tl.dot(p.to(V.dtype), V)
```

Important shapes:

- `M`: `[num_queries_per_kv_padded]`
- `L`: `[num_queries_per_kv_padded]`
- `p`: `[num_queries_per_kv_padded, BLOCK_SIZE]`
- `acc`: `[num_queries_per_kv_padded, HEAD_SIZE_PADDED]`

So `acc` is accumulated in head-dimension space, not token space.

## Final normalization

After all tiles are processed:

```python
acc = acc / (L[:, None] + 1e-10)
```

Then, if FP8 output is requested, the result is rescaled and clamped before
being written to `output`.




