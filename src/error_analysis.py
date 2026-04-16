"""
Qualitative error analysis helpers.

Produces a small, printable inventory of the model's mistakes alongside a few
correct predictions, intended to surface the morphological and discourse-level
error patterns discussed in the report.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from datasets import Dataset
from transformers import Trainer

from .config import LABELS


@dataclass
class PredictionRecord:
    index: int
    premise: str
    hypothesis: str
    true_label: str
    predicted_label: str
    correct: bool


def collect_predictions(trainer: Trainer, dataset: Dataset) -> List[PredictionRecord]:
    """Return a list of :class:`PredictionRecord` for every row in ``dataset``."""
    raw_pred, _, _ = trainer.predict(dataset)
    predictions = np.argmax(raw_pred, axis=1)
    labels = dataset["label"]

    records: List[PredictionRecord] = []
    for i, (pred, true) in enumerate(zip(predictions, labels)):
        records.append(
            PredictionRecord(
                index=i,
                premise=dataset[i]["premise"],
                hypothesis=dataset[i]["hypothesis"],
                true_label=LABELS[true],
                predicted_label=LABELS[int(pred)],
                correct=bool(pred == true),
            )
        )
    return records


def summarise(
    records: List[PredictionRecord],
    dataset_name: str,
    num_errors: int = 5,
    num_successes: int = 3,
) -> None:
    """Print a compact summary of errors and successes for a split."""
    errors = [r for r in records if not r.correct]
    successes = [r for r in records if r.correct]

    header = f"=== {dataset_name} ==="
    print(f"\n{header}")
    print(
        f"Total: {len(records)}  |  Correct: {len(successes)}  |  "
        f"Errors: {len(errors)}  |  Accuracy: {len(successes) / len(records):.4f}"
    )

    print(f"\nFirst {num_errors} errors:")
    for r in errors[:num_errors]:
        _print_record(r)

    print(f"\nFirst {num_successes} correct predictions:")
    for r in successes[:num_successes]:
        _print_record(r)


def _print_record(r: PredictionRecord) -> None:
    print(f"  [{r.index}]")
    print(f"    Premise:    {r.premise}")
    print(f"    Hypothesis: {r.hypothesis}")
    print(f"    Gold: {r.true_label}  |  Predicted: {r.predicted_label}")
