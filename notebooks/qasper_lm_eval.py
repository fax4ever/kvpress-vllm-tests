"""Connect the standard lm-eval SCROLLS/Qasper task to our compression runs."""

from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys
import time
import traceback
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.loggers import EvaluationTracker
from lm_eval.tasks.scrolls.task import Qasper, _drop_duplicates_in_input


class _Tee:
    """Keep notebook output visible and flush a readable copy to disk."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text.replace("\r", "\n"))
        self.log.flush()
        return self.stream.write(text)

    def flush(self):
        self.log.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def evaluation_log(output_dir):
    """Save notebook prints, stderr progress bars, and exceptions for this run."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    started = time.perf_counter()
    with path.open("w", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print(f"[{datetime.now().isoformat(timespec='seconds')}] Run started. Log: {path}")
            try:
                yield path
            except BaseException:
                print(f"[{datetime.now().isoformat(timespec='seconds')}] Run interrupted or failed")
                traceback.print_exc()
                raise
            else:
                print(f"[{datetime.now().isoformat(timespec='seconds')}] Run finished in {time.perf_counter() - started:.0f}s")


class QasperTask(Qasper):
    """Use the direct Hugging Face loader supported by datasets 3.6.0."""

    EVAL_HARNESS_NAME = "scrolls_qasper"

    def __init__(self):
        super().__init__()
        self.set_config("task", "scrolls_qasper")

    def download(self, *args, **kwargs):
        self.dataset = load_dataset(
            "tau/scrolls", "qasper", trust_remote_code=True,
        )
        for split in ("train", "validation"):
            self.dataset[split] = _drop_duplicates_in_input(self.dataset[split])


def kvpress_model(*, press=None, max_gen_toks=256, **kwargs):
    """Use the harness HF backend; activate a fresh press scope per model call."""
    from lm_eval.models.huggingface import HFLM

    class KVPressHFLM(HFLM):
        @property
        def max_gen_toks(self):
            return max_gen_toks

        def _model_call(self, *args, **call_kwargs):
            with press(self.model) if press is not None else nullcontext():
                return super()._model_call(*args, **call_kwargs)

        def _model_generate(self, *args, **generation_kwargs):
            with press(self.model) if press is not None else nullcontext():
                return super()._model_generate(*args, **generation_kwargs)

    # Presses maintain per-sequence state, so do not batch independent sequences.
    return KVPressHFLM(batch_size=1, **kwargs)


def evaluate_qasper(model, *, output_dir, fraction=1.0, model_args=None, metadata=None):
    """Run and save the standard protocol through the harness APIs."""
    task = QasperTask()
    if metadata:
        task.set_config("metadata", metadata, update=True)
    tracker = EvaluationTracker(output_path=str(output_dir))
    report = simple_evaluate(
        model=model,
        tasks=[task],
        model_args=model_args,
        num_fewshot=0,
        limit=fraction if fraction < 1 else None,
        apply_chat_template=True,
        log_samples=True,
        evaluation_tracker=tracker,
        bootstrap_iters=0,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )
    samples = report.pop("samples")
    tracker.save_results_aggregated(results=report, samples=samples)
    for task_name, task_samples in samples.items():
        tracker.save_results_samples(task_name=task_name, samples=task_samples)
    return report
