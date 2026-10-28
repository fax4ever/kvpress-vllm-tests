# Thesis results

This directory contains the benchmark results presented in the thesis.

## Benchmark data

Notebooks 07 through 12 save the full selected dataset splits (`FRACTION = 1`):

- [kvpress_ruler](kvpress_ruler/) and [vllm_ruler](vllm_ruler/): notebooks 07 and 08.
- [kvpress_qasper_lm_eval](kvpress_qasper_lm_eval/) and [vllm_qasper_lm_eval](vllm_qasper_lm_eval/): notebooks 09 and 10.
- [kvpress_longbench_lm_eval](kvpress_longbench_lm_eval/) and [vllm_longbench_lm_eval](vllm_longbench_lm_eval/): notebooks 11 and 12.

RULER saves predictions in `predictions.csv` and scores in `metrics.json`.
Qasper and LongBench save native harness result JSON and per-example JSONL files.

For RULER, each framework has 500 examples per task, 13 tasks, two context
lengths (4096 and 8192), and nine configurations, totaling 117,000 predictions.

## Benchmark figures

The comparison notebooks read the saved metrics above:

- Notebook 13: [RULER 4096 heatmaps](compare_ruler_4096_heatmaps.png),
  [RULER 8192 heatmaps](compare_ruler_8192_heatmaps.png), and
  [RULER average score versus compression ratio](compare_ruler_score_vs_ratio.png).
  Heatmaps show per-task string-match scores. The average-score plot averages
  over all 13 tasks and both context lengths.
- Notebook 14: [Qasper scores](compare_qasper_scores.png), showing F1.
- Notebook 15: [LongBench scores](compare_longbench_lm_eval_scores.png), showing
  ROUGE-L for `gov_report` and F1 for `hotpotqa`, plus per-task heatmaps:
  [GovReport](compare_longbench_gov_report_heatmaps.png) and
  [HotpotQA](compare_longbench_hotpotqa_heatmaps.png).

These figures compare both algorithms and frameworks with their uncompressed baselines.

## Superseded LongBench results

`kvpress_longbench/`, `vllm_longbench/`, and
`compare_longbench_scores.png` retain the earlier custom evaluation artifacts.
They have known prompting and thinking-mode errors and must not be used as
results of the corrected protocol. Replacement Notebooks 11, 12, and 15 do not
read them. The current notebooks use the harness’s built-in LongBench tasks unchanged
and log example requests so their prompts can be inspected.
