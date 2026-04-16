"""
Evaluation utilities for language-disaggregated XNLI accuracy reporting.
"""

from typing import Dict

from datasets import Dataset
from transformers import Trainer


def evaluate_per_language(
    trainer: Trainer, datasets: Dict[str, Dataset]
) -> Dict[str, float]:
    """Run ``trainer.evaluate`` on each named dataset and return accuracies.

    Parameters
    ----------
    trainer : Trainer
        A fine-tuned ``Trainer``.
    datasets : dict[str, Dataset]
        Mapping from label (e.g. ``"en"``, ``"tr"``, ``"mixed"``) to a
        tokenized dataset ready for evaluation.

    Returns
    -------
    dict[str, float]
        ``{label: accuracy}`` for each input dataset.
    """
    results = {}
    for label, dataset in datasets.items():
        metrics = trainer.evaluate(dataset)
        results[label] = metrics["eval_accuracy"]
    return results


def pretty_print_results(results: Dict[str, float], title: str = "Results") -> None:
    """Print a small, aligned table of per-language accuracies."""
    print(f"\n{title}")
    print("-" * len(title))
    for label, acc in results.items():
        print(f"  {label:<10s} {acc:.4f}")
