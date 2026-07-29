# kvpress-vllm-tests

Comparing KV cache compression strategies for long-context LLM inference,
with a focus on bringing techniques like KeyDiffPress into vLLM.

## Context

This repository is part of a master thesis exploring whether KV cache
compression — specifically the KeyDiff algorithm
([arXiv:2504.15364](https://arxiv.org/abs/2504.15364)) — can be applied
to [vLLM](https://github.com/vllm-project/vllm), a high-throughput
inference engine built around PagedAttention.

Today, KV cache compression and production serving engines live in
separate worlds. Libraries like
[kvpress](https://github.com/NVIDIA/kvpress) (NVIDIA) implement
compression algorithms on top of Hugging Face Transformers, but they
don't integrate with the paged memory management and continuous batching
that make vLLM fast. On the other side, vLLM provides excellent serving
performance but has no built-in support for KV cache compression.

The thesis goal is to bridge this gap: understand both systems deeply
enough to evaluate whether KeyDiff-style compression can work within
vLLM's architecture, and if so, prototype it. This is a hard problem —
vLLM's KV cache is physically paged, non-contiguous, and managed by a
block allocator that compression algorithms were never designed for.

## What's in This Repo

The work is organized in two parts.

### Notebooks — NIAH Comparison (`notebooks/`)

The first part is a Needle-in-a-Haystack (NIAH) benchmark that compares
kvpress and vLLM side by side using Qwen3-8B on an NVIDIA A10G GPU.
The NIAH test inserts a known sentence ("the needle") at various depth
positions within a long document, then asks the model to retrieve it.
This measures whether compression degrades the model's ability to attend
to information at different positions in the context.

The notebooks are numbered sequentially:

| Notebook | Description |
|----------|-------------|
| `00_setup_check` | Verifies GPU access, installs dependencies, runs smoke tests |
| `01_kvpress_niah` | Runs NIAH with KeyDiffPress at compression ratios 0%, 25%, 50%, 75% |
| `02_vllm_niah` | Runs the same NIAH benchmark on vLLM (no compression baseline) |
| `02a_vllm_fork_setup` | Prepares a [vLLM fork](https://github.com/fax4ever/vllm) for testing Python-level modifications |
| `03_compare_results` | Loads results from both frameworks, produces heatmaps and comparison charts |

The kvpress configuration uses **PrefillDecodingPress**, combining
BlockPress(KeyDiffPress) for prefill-phase compression with
CompressionRatioDecodingPress(KeyDiffPress) for decoding-phase
compression. Results (metrics and predictions) are saved under
`notebooks/results/`.

### vLLM Experiments (`vllm-experiments/`)

The second part is a collection of smaller, focused notebooks that
explore individual pieces of vLLM's internals. The goal here is to
build understanding incrementally — testing isolated functions and
primitives before attempting any larger integration.

The first experiments focus on **scatter/gather operations** for paged
KV caches, based on work in the
[kv-press branch](https://github.com/vllm-project/vllm/compare/main...fax4ever:vllm:kv-press)
of the vLLM fork. These operations are fundamental to any compression
scheme that needs to read KV cache pages into a contiguous buffer,
apply compression, and write the results back.

## Environment

The notebooks are designed to run on an **OpenShift AI Workbench** with:

- **Image:** PyTorch (CUDA)
- **GPU:** NVIDIA A10G (24 GB VRAM) or similar with 20+ GB VRAM
- **Container size:** Large (8+ CPU, 32+ GB RAM)

Key dependencies: vLLM 0.18, kvpress 0.5.4, Transformers 4.57,
Qwen3-8B. Full install instructions are in notebook `00_setup_check`.

## Future Directions

The long-term research direction is to implement KV cache compression
directly inside vLLM. This would require solving several hard problems:
gathering non-contiguous paged KV entries into a dense buffer, running
a compression pass (like KeyDiff's key-similarity scoring), and
scattering the survivors back into the page table — all without breaking
vLLM's block allocator, scheduler, or continuous batching guarantees.

This is very much research-stage work. The `vllm-experiments/` notebooks
are the first steps toward understanding the building blocks well enough
to attempt it.

## References

- S. Dong et al. *QPress and KeyDiff: Scalable KV Cache Compression via Query-agnostic Strategies.* [arXiv:2504.15364](https://arxiv.org/abs/2504.15364), 2025.
- G. Kamradt. *Needle in a Haystack — Pressure Testing LLMs.* [GitHub](https://github.com/gkamradt/LLMTest_NeedleInAHaystack), 2023.
- N. F. Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024.
- [kvpress](https://github.com/NVIDIA/kvpress) — NVIDIA's KV cache compression library
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput LLM serving engine
- [vLLM fork (kv-press branch)](https://github.com/vllm-project/vllm/compare/main...fax4ever:vllm:kv-press) — Scatter/gather experiments

## License

[Apache 2.0](LICENSE)
