# KV Cache, Prefix Caching, and Compression in vLLM

## 1. Requests and the KV cache

A **request** in vLLM is a single inference call: prompt in, completion
out (prefill → decode → done). In a multi-turn conversation, each turn
is a separate request — the application concatenates previous turns into
the new prompt.

The KV cache is request-scoped by default. When a request completes,
its blocks return to the free pool.

## 2. Prefix caching

Automatic Prefix Caching (enabled by default in vLLM V1) retains blocks
after a request completes. When a new request shares the same prefix
tokens, it reuses the cached blocks instead of re-prefilling.

The hash is computed from **token IDs**, not from the K/V values:

```
hash = hash_function((parent_block_hash, token_ids_in_block, extra_keys))
```

It is a chain hash — each block's hash includes its parent's, encoding
the exact token sequence and position. Only fully-filled blocks are
cached (`num_full_blocks = num_tokens // block_size`). A block becomes
shared (`ref_cnt >= 2`) when a second request matches its hash.

This is essential for production multi-turn serving. Without it, every
turn re-prefills the entire conversation history.

## 3. Compression and prefix caching

The current implementation (branch `fax-0.18`) force-disables prefix
caching for both algorithms. However, **compressed blocks are reusable**
and compression is compatible with prefix caching.

### Why compressed blocks are reusable

The hash matches because it is based on token IDs, not K/V values.
Due to causal attention, token i's K/V depends only on tokens 0..i —
the K/V values of the survivors are correct regardless of which request
computed them. A subsequent request matching the hash gets the
compressed blocks and starts from that state:

```
Request 1: hash(token_ids) => pages (compressed after prefill)
Request 2: hash(token_ids) => same hash => gets compressed pages
            → starts from compressed state, continues from there
```

### What needs to change

The force-disable is unnecessary. Making it work requires small,
targeted changes:

1. **Trim the hash chain after compression.** Invalidate hashes for
   blocks beyond the compressed length (a short loop).
2. **Attach `num_kv_discarded` to cached blocks.** One integer so
   the reusing request knows the compression state.
3. **Don't re-compress already-compressed data.** The existing
   `kv_compressed` flag already handles this.

## 4. Scope

This analysis identifies that KV compression is compatible with prefix
caching and outlines what would need to change. The actual
implementation of multi-request conversation support (prefix caching
with compressed blocks) is out of scope for this thesis — it is future
work, not a blocker for evaluating the compression algorithms
themselves.

The thesis focuses on single-request compression: evaluating
the quality and memory impact of both algorithms (full replacement
and filtering) within a single request lifecycle.

## 5. Sources

- [Automatic Prefix Caching — vLLM docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [KV-Cache Wins You Can See — llm-d blog](https://llm-d.ai/blog/kvcache-wins-you-can-see)
- [Prefix Caching in vLLM & SGLang (2026)](https://llm-academy.dev/optimization/prefix-caching/)
- [How prompt caching works — sankalp's blog](https://sankalp.bearblog.dev/how-prompt-caching-works/)
