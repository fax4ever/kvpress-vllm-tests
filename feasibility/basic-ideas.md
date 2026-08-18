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
   physical locations in the paged cache. Each entry is a global slot
   index: `physical_block_id * block_size + offset`, addressing an
   absolute position in the cache. The tensor only covers the tokens
   being operated on (e.g. a single sequence's context, or newly
   appended tokens), not the entire cache.

3. **Block Allocator** — holds the global view of paged memory: which
   physical blocks are free, which are assigned to which sequences, and
   reference counts for sharing. The block allocator assigns blocks and
   builds the block tables that slot mappings are derived from.

## Primitives

0. **Create KV cache** — allocate the paged cache tensors. Flash
   Attention layout: `[num_blocks, block_size, num_kv_heads, head_size]`
   for both keys and values.

1. **Scatter** — write dense K/V tensors into the paged cache.
   Implemented by `ops.reshape_and_cache_flash`: takes the dense tensors
   and the `slot_mapping`, writes each token to its designated slot.
   The `slot_mapping` must have one entry per token in the dense tensor,
   and each entry must reference a slot assigned to the sequence by
   the block allocator — shape and assignment consistency are the
   caller's responsibility.

2. **Gather** — recover a sequence's K/V from the paged cache into
   dense tensors. Reverses the slot index back into block and offset:
   `block_idx = slot // block_size`, `offset = slot % block_size`,
   then indexes directly: `cache[block_idx, offset]`. Not required by
   attention computation, since the Flash Attention kernel reads
   directly from pages via the block table. We add it for compression,
   which needs the full sequence as a contiguous tensor for scoring.

3. **Score per head** — score each token independently per KV head,
   producing a relevance ranking. Operates on dense tensors produced
   by gather. The compression algorithm lives here (e.g. KeyDiff's
   key-similarity metric). Returns `[num_kv_heads, seq_len]`.

4. **Select per head** — keep a subset of tokens per KV head based on
   score rankings. Each head may retain a different set of positions.
   Returns dense `[compacted_len, num_kv_heads, head_size]` tensors.
   Used by the total-replacement strategy.

5. **Filter new token** — decide per head whether the newest token
   survives filtering. Computes a top-k threshold from the scores and
   checks whether the new token's score is above it. Used by the
   append-only strategy (FilteringPress) for online keep/skip decisions
   at decode time.

## Compression Strategies

Two strategies for reducing the KV cache, each composing the primitives
above differently:

1. **Total replacement** (compact) — compress an entire cached sequence
   at once: gather → score → select → scatter back into the first
   `compacted_len` slots. The cache shrinks; the block table stays the
   same but the sequence length tracked by the scheduler decreases.

2. **Append-only filtering** (FilteringPress) — compress during
   generation, one token at a time: gather → score (cached + new
   token) → filter → conditionally scatter the new token. The cache
   only grows, never shrinks — compression comes from skipping
   low-scoring tokens before they enter the cache.