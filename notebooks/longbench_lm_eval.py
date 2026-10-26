"""Run the built-in LongBench tasks through LM Evaluation Harness."""

import logging
import sys

from lm_eval import simple_evaluate
from lm_eval.loggers import EvaluationTracker


def evaluate_longbench(model, *, tasks, output_dir, fraction=1.0, model_args=None, metadata=None):
    """Run the harness unchanged, log example requests, and save native results."""
    for task in tasks:
        if metadata:
            task.set_config("metadata", metadata, update=True)
    tracker = EvaluationTracker(output_path=str(output_dir))
    # Send full request traces to the tee's file, keeping notebook output concise.
    logger = logging.getLogger("lm_eval.evaluator_utils")
    handler = logging.StreamHandler(getattr(sys.stdout, "log", sys.stdout))
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        report = simple_evaluate(
            model=model, tasks=tasks, model_args=model_args,
            num_fewshot=0, limit=fraction if fraction < 1 else None,
            apply_chat_template=True, log_samples=True, write_out=True,
            evaluation_tracker=tracker, bootstrap_iters=0,
            random_seed=42, numpy_random_seed=42, torch_random_seed=42, fewshot_random_seed=42,
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
        handler.close()
    samples = report.pop("samples")
    tracker.save_results_aggregated(results=report, samples=samples)
    for task_name, task_samples in samples.items():
        tracker.save_results_samples(task_name=task_name, samples=task_samples)
    return report
