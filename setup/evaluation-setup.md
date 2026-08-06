# Evaluation Setup

**Model:** `Qwen/Qwen3-8B` — `flash_attention_2`, no quantization

## Methods Under Test

Four decoding-phase compression variants using **KeyDiffPress** as base scorer:

| Registry Name | Press |
|---------------|-------|
| `compression_ratio_decoding_keydiff` | `CompressionRatioDecodingPress` |
| `uniform_filtering_keydiff` | `UniformFilteringPress` |
| `filtering_zerofill_keydiff` | `FilteringPress` |
| `filtering_stale_keydiff` | `FilteringPress(fill_padding=False)` |

## RULER (notebook 06)

[kvpress evaluation framework](https://github.com/NVIDIA/kvpress/tree/main/evaluation) —
same setup as the [kvpress leaderboard](https://huggingface.co/spaces/nvidia/kvpress-leaderboard).

- [RULER](https://huggingface.co/datasets/simonjegou/ruler) 4096, string match, compression ratios: 0.25, 0.50, 0.75

## NIAH (notebook 02)

Custom loop, compared against vLLM baseline (notebook 04).

- [Paul Graham Essays](https://huggingface.co/datasets/alessiodevoto/paul_graham_essays), ROUGE-L
- Compression ratios: 0.0, 0.25, 0.50, 0.75 — depths: 0/25/50/75/100% — context: 4096, 8192
