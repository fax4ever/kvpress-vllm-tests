# Prefill attention in `prefix_prefill.py`

This note focuses on the prefill path of the Triton kernel in
`vllm/v1/attention/ops/prefix_prefill.py`, specifically the `_fwd_kernel`
(non-ALiBi case).

The companion decode document covers `kernel_paged_attention_2d` in
`chunked_prefill_paged_decode.py`.

## How prefill is invoked

In `chunked_prefill_paged_decode.py`, the entry point
`chunked_prefill_paged_decode()` dispatches prefill when the batch contains
at least one multi-token query:

```python
if max_query_len > 1:
    context_attention_fwd(
        q=query, k=key, v=value, o=output,
        kv_cache_dtype=kv_cache_dtype,
        k_cache=key_cache, v_cache=value_cache,
        b_loc=block_table, b_start_loc=query_start_loc,
        b_seq_len=seq_lens, ...,
        skip_decode=True,
    )
```

With `skip_decode=True`, the kernel early-returns for any sequence with
`query_len == 1`, processing only multi-token (prefill) sequences.

## Two-phase attention

The key structural difference from decode: each prefill sequence has both
**cached context** (already in the KV cache from previous iterations) and
**new request tokens** (the current query's K/V, not yet cached).

The kernel splits attention into two sequential phases:

1. **Context phase** — query attends to cached KV (no causal mask)
2. **Request phase** — query attends to the current request's own K/V (causal mask applied)

The total `seq_len` for a sequence is decomposed as:

```
seq_len = ctx_len + query_len
```

where:

```python
cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len
```

For a pure prefill (first forward pass), `ctx_len == 0` and the context
phase loop body never executes.

## Kernel grid

```python
grid = (batch, head, triton.cdiv(max_input_len, BLOCK_M))
```

- `program_id(0)` = sequence index in the batch
- `program_id(1)` = query head index
- `program_id(2)` = query tile index (`start_m`)

Each program instance computes `BLOCK_M` rows of the output for one query
head in one sequence. This is fundamentally different from decode, where
the grid is `(num_seqs, num_kv_heads)` and GQA is handled inside the
kernel.

For prefill, GQA is handled by mapping the query head to its KV head:

```python
cur_kv_head = cur_head // num_queries_per_kv
```

## Input tensor layouts

The kernel receives both raw tensors and cache tensors:

- `Q`, `K`, `V`: `[total_tokens, num_heads, head_size]` — packed across
  all sequences in the batch (no padding between sequences)
- `K_cache`, `V_cache`: paged KV cache (see below)
- `B_Loc`: `[num_seqs, max_num_blocks_per_seq]` — block table mapping
  logical blocks to physical blocks
- `B_Start_Loc`: `[num_seqs + 1]` — cumulative token offsets into Q/K/V
- `B_Seqlen`: `[num_seqs]` — total sequence length (context + query)

## KV cache layout

Same as in the decode path:

- `K_cache`: `[num_blocks, num_kv_heads, head_size // x, block_size, x]`
- `V_cache`: `[num_blocks, num_kv_heads, head_size, block_size]`

where `x` is the packed inner dimension for keys.

## Query loading

Each program instance loads a tile of `BLOCK_M` query positions:

```python
offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
off_q = (
    (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
    + cur_head * stride_qh
    + offs_d[None, :] * stride_qd
)
q = tl.load(Q + off_q,
            mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
            other=0.0)
```

Shape of `q`: `[BLOCK_M, BLOCK_DMODEL_PADDED]`.

The query is loaded once and reused across all KV tiles in both phases.

## Phase 1: Context (cached KV)

The context loop iterates over tiles of `BLOCK_SIZE` cached tokens:

```python
for start_n in tl.range(0, cur_batch_ctx_len, BLOCK_SIZE,
                        loop_unroll_factor=num_unroll_cache):
```

### Loading K from cache (5D addressing)

```python
token_indices = start_n + offs_bs_n
bn_logical_indices = token_indices // PHYSICAL_BLOCK_SIZE
bn = tl.load(B_Loc + cur_batch * stride_b_loc_b
             + bn_logical_indices * stride_b_loc_s).to(tl.int64)
internal_offsets = token_indices % PHYSICAL_BLOCK_SIZE

off_k = (
    bn[None, :] * stride_k_cache_bs
    + cur_kv_head * stride_k_cache_h
    + (offs_d[:, None] // x) * stride_k_cache_d
    + internal_offsets[None, :] * stride_k_cache_bl
    + (offs_d[:, None] % x) * stride_k_cache_x
)
```

The addressing is the same as in the decode kernel: each token index is
translated into a physical block ID (`bn`) and an offset within that block
(`internal_offsets`).

Broadcasting:

- `bn`: `[BLOCK_SIZE]`, expanded to `[1, BLOCK_SIZE]` via `bn[None, :]`
- `offs_d`: `[BLOCK_DMODEL_PADDED]`, expanded to `[BLOCK_DMODEL_PADDED, 1]`
  via `offs_d[:, None]`
- `internal_offsets`: `[BLOCK_SIZE]`, expanded to `[1, BLOCK_SIZE]`

Result: `off_k` is a `[BLOCK_DMODEL_PADDED, BLOCK_SIZE]` address grid.

K tile shape: `[BLOCK_DMODEL_PADDED, BLOCK_SIZE]` (head_dim x tokens,
transposed for the dot product).

### Loading V from cache (4D addressing)

```python
off_v = (
    bn[:, None] * stride_v_cache_bs
    + cur_kv_head * stride_v_cache_h
    + offs_d[None, :] * stride_v_cache_d
    + internal_offsets[:, None] * stride_v_cache_bl
)
```

V tile shape: `[BLOCK_SIZE, BLOCK_DMODEL_PADDED]` (tokens x head_dim).

### Attention computation (context)

```python
qk = sm_scale * tl.dot(q, k, input_precision=IN_PRECISION)
qk = tl.where((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len,
              qk, float("-inf"))
```

No causal mask is applied in the context phase: all context tokens are
in the past relative to every query token.

Sliding window is applied if enabled:

```python
if SLIDING_WINDOW > 0:
    qk = tl.where(
        (cur_batch_ctx_len + offs_m[:, None])
        - (start_n + offs_bs_n[None, :]) < SLIDING_WINDOW,
        qk, float("-inf"))
```

The `offs_m` positions are shifted by `ctx_len` to get absolute positions
in the sequence, and tokens beyond the window are masked out.

Then online softmax accumulation (same algorithm as decode):

```python
m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
p = tl.exp(qk - m_ij[:, None])
p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
l_ij = tl.sum(p, axis=1)
alpha = tl.exp(m_i - m_ij)
alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
acc = acc * alpha[:, None]
acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
l_i = l_i * alpha + l_ij
m_i = m_ij
```

Important shapes:

- `m_i`, `l_i`: `[BLOCK_M]` — one value per query position in the tile
- `qk`, `p`: `[BLOCK_M, BLOCK_SIZE]`
- `acc`: `[BLOCK_M, BLOCK_DMODEL_PADDED]`

Note that Triton's `tl.dot(p, v, acc=acc)` fuses the multiply-add into
the accumulator.

## Phase 2: Request (new K/V tokens)

After the context loop, the kernel computes attention against the current
request's own K and V tokens. These come from the raw `K`/`V` tensors
(not the cache).

```python
off_k = (
    offs_n[None, :] * stride_kbs
    + cur_kv_head * stride_kh
    + offs_d[:, None] * stride_kd
)
k_ptrs = K + off_k
v_ptrs = V + off_v
```

The request loop upper bound depends on the attention mode:

```python
if CAUSAL:
    key_range_upper = block_mask * (start_m + 1) * BLOCK_M
else:
    q_len_pad = (cur_batch_query_len + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    key_range_upper = block_mask * q_len_pad
```

For causal attention, each query tile only needs to attend to KV positions
up to the end of the current tile (upper-triangular structure). For
non-causal (e.g., DFlash drafting), the range covers all query tokens.

### Loading K and V from raw tensors

```python
k = tl.load(
    k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
    mask=dim_mask[:, None]
    & ((start_n + offs_n[None, :]) < cur_batch_query_len),
    other=0.0)
```

K tile shape: `[BLOCK_DMODEL_PADDED, BLOCK_N]`.
V tile shape: `[BLOCK_N, BLOCK_DMODEL_PADDED]`.

These are loaded directly from the packed Q/K/V tensors using the
sequence's start offset (`cur_batch_in_all_start_index`).

### Causal masking (request)

```python
if CAUSAL:
    attn_mask = valid_kv & (offs_m[:, None] >= (start_n + offs_n[None, :]))
else:
    attn_mask = valid_kv

qk = tl.where(attn_mask, qk, float("-inf"))
```

For causal attention, query position `m` can only attend to key
positions `n <= m` within the request. This is the standard lower-triangular
mask. Combined with the loop bound of `(start_m + 1) * BLOCK_M`, this
makes each query tile skip later KV tiles entirely.

### Online softmax (request)

Identical algorithm to the context phase, continuing the running `m_i`,
`l_i`, and `acc` from where the context phase left off.

## Final normalization

After both phases:

```python
acc = acc / (l_i[:, None] + 1e-10)
```

Then, if FP8 output is requested:

```python
if USE_FP8:
    acc = acc * tl.load(out_scale_inv)
    acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
```

The result is written to the output tensor:

```python
off_o = (
    (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
    + cur_head * stride_oh
    + offs_d[None, :] * stride_od
)
tl.store(out_ptrs, acc,
         mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len))
```

## Comparison with decode

| Aspect | Prefill (`_fwd_kernel`) | Decode (`kernel_paged_attention_2d`) |
|--------|------------------------|--------------------------------------|
| Grid axes | `(batch, query_head, query_tile)` | `(seq, kv_head)` |
| GQA handling | One query head per program; KV head derived via `cur_head // num_queries_per_kv` | One KV head per program; all its query heads processed together (`num_queries_per_kv_padded`) |
| KV source | Two sources: paged cache (context) + raw K/V tensors (request) | Single source: paged cache only |
| Tiling dimension | Tiles over query (M) and KV (N) separately | Tiles over KV (N) only; query is a single token |
| Causal mask | Applied in request phase (lower-triangular) | Not needed (single query token always attends to all past) |
| Accumulator shape | `[BLOCK_M, BLOCK_DMODEL_PADDED]` — multiple query positions | `[num_queries_per_kv_padded, HEAD_SIZE_PADDED]` — one token, multiple GQA heads |
| Loop structure | Two sequential loops (context + request) | Single loop over all cached KV |
