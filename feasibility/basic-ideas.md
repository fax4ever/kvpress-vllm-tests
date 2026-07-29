# Basic Ideas

## Flash Attention 2

We target only `AttentionBackendEnum.FLASH_ATTN` — the Flash Attention 2
backend used on NVIDIA GPUs via CUDA. This is what vLLM selects by
default for standard models (like Qwen3-8B) on Ampere and Hopper GPUs.

## Terminology

1. **Sequence** — a batch can process multiple sequences in parallel.
   Each sequence is independent: its own context, block table, and
   sequence length.    Each physical block is assigned to exactly one
   sequence — sequences never share cache memory unless they
   share the same prefix tokens (and thus identical K/V values).

2. **`slot_mapping`** — a flat tensor mapping token positions to
   physical locations in the paged cache. Each entry is a slot index:
   `physical_block_id * block_size + offset`. The slot mapping provides
   a local view — it only covers the tokens being operated on, not the
   entire cache.

3. **Block Allocator** — holds the global view of paged memory: which
   physical blocks are free, which are assigned to which sequences, and
   reference counts for sharing. The block allocator assigns blocks and
   builds the block tables that slot mappings are derived from.

## Operations

0. **Create KV cache** — allocate the paged cache tensors. Flash
   Attention layout: `[num_blocks, block_size, num_kv_heads, head_size]`
   for both keys and values.

1. **Scatter** — write dense K/V tensors into the paged cache.
   Implemented by `ops.reshape_and_cache_flash`: takes the dense tensors
   and the `slot_mapping`, writes each token to its designated slot.

2. **Gather** — recover a sequence's K/V from the paged cache into
   dense tensors. Pure PyTorch indexing using `slot_mapping`:
   `cache[block_idx, offset]`. Not required by attention computation,
   since the Flash Attention kernel reads directly from pages via the
   block table. We add it for compression, which needs the full
   sequence as a contiguous tensor for scoring.

3. **Compact** — the full compression primitive, applied to a single
   sequence: gather → score/select → scatter back into the first
   `compacted_len` slots.