# Thesis results

This directory contains the benchmark results presented in the thesis.

## Benchmark data

Notebooks 07 through 12 save the full selected dataset splits (`FRACTION = 1`):

- [kvpress_ruler](kvpress_ruler/) and [vllm_ruler](vllm_ruler/): notebooks 07 and 08.
- [kvpress_qasper](kvpress_qasper/) and [vllm_qasper](vllm_qasper/): notebooks 09 and 10.
- [kvpress_longbench](kvpress_longbench/) and [vllm_longbench](vllm_longbench/): notebooks 11 and 12.

Each directory contains `predictions.csv` with reference answers, generated
answers, and run settings, and `metrics.json` with benchmark scores.
For RULER, each framework has 500 examples per task, 13 tasks, two context
lengths (4096 and 8192), and nine configurations, totaling 117,000 predictions.

## Benchmark figures

The comparison notebooks read the saved metrics above:

- Notebook 13: [RULER 4096 heatmaps](compare_ruler_4096_heatmaps.png),
  [RULER 8192 heatmaps](compare_ruler_8192_heatmaps.png), and
  [RULER average score versus compression ratio](compare_ruler_score_vs_ratio.png).
  Heatmaps show per-task string-match scores. The average-score plot averages
  over all 13 tasks and both context lengths.
- Notebook 14: [Qasper scores](compare_qasper_scores.png), showing F1 and exact match.
- Notebook 15: [LongBench scores](compare_longbench_scores.png), showing
  ROUGE-L for `gov_report` and F1 for `hotpotqa`.

These figures compare both algorithms and frameworks with their uncompressed baselines.
