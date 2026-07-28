"""
Shared evaluation utilities for Needle-in-a-Haystack (NIAH) experiments.

Replicates the NIAH benchmark approach from kvpress:
- Paul Graham essays as haystack filler
- Token-level needle insertion at configurable depths
- ROUGE scoring between needle text and predicted answer

Dataset schema follows kvpress conventions:
  context, question, answer_prefix, answer, max_new_tokens, task
"""

import logging
from typing import Optional

import pandas as pd
from rouge_score import rouge_scorer as rs
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

_scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def insert_needle_in_haystack(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    max_context_length: int,
    needle_depth: int | list[int],
    context_wrapper: str = "This is a very long story book: <book> {context} </book>.",
    needle_text: Optional[str] = None,
    answer_prefix: Optional[str] = None,
    question_text: Optional[str] = None,
) -> pd.DataFrame:
    """
    Insert a needle sentence into the haystack context at specified depth(s).

    Tokenizes the haystack, truncates to fit within max_context_length,
    inserts the needle at each requested depth percentage, and returns
    one row per depth.

    Replicates kvpress/evaluation/benchmarks/needle_in_haystack/utils.py.
    """
    original_context = df["context"].iloc[0]
    needle_text = needle_text or df["needle"].iloc[0]
    question_text = question_text or df["question"].iloc[0]
    answer_prefix = answer_prefix or df["answer_prefix"].iloc[0]
    max_new_tokens = df["max_new_tokens"].iloc[0]

    logger.info(f"Preparing NIAH dataset. Needle: {needle_text}")

    tokenized_needle = tokenizer.encode(needle_text, add_special_tokens=False)
    context_length_limit = max_context_length - len(tokenized_needle) - 150
    tokenized_context = tokenizer.encode(original_context, add_special_tokens=False)[:context_length_limit]

    needle_depth = [needle_depth] if isinstance(needle_depth, int) else needle_depth
    new_rows = []

    for depth in needle_depth:
        needle_index = int(len(tokenized_context) * depth / 100)
        new_tokenized_context = (
            tokenized_context[:needle_index]
            + tokenized_needle
            + tokenized_context[needle_index:]
        )
        decoded_context = tokenizer.decode(new_tokenized_context, skip_special_tokens=True)
        final_context = context_wrapper.format(context=decoded_context)
        new_rows.append({
            "context": final_context,
            "needle": needle_text,
            "needle_depth": depth,
            "question": question_text,
            "answer_prefix": answer_prefix,
            "max_new_tokens": max_new_tokens,
            "task": "needle_in_haystack",
        })

    return pd.DataFrame(new_rows)


def calculate_niah_metrics(df: pd.DataFrame) -> list[dict]:
    """
    Score NIAH predictions using ROUGE between needle text and predicted answer.

    Replicates kvpress/evaluation/benchmarks/needle_in_haystack/calculate_metrics.py.
    Uses rouge-score (Google) instead of rouge package, same underlying algorithm.
    """
    scores = []
    for _, row in df.iterrows():
        needle = row["needle"].strip()
        prediction = row["predicted_answer"].strip()
        if not prediction:
            scores.append({"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}})
            continue
        result = _scorer.score(needle, prediction)
        scores.append({
            "rouge-1": {"f": result["rouge1"].fmeasure},
            "rouge-2": {"f": result["rouge2"].fmeasure},
            "rouge-l": {"f": result["rougeL"].fmeasure},
        })
    return scores


def rouge_l_f_scores(metrics: list[dict]) -> list[float]:
    """Extract ROUGE-L F1 scores from calculate_niah_metrics output."""
    return [m["rouge-l"]["f"] for m in metrics]
