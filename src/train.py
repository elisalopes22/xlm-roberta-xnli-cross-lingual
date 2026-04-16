"""
Fine-tuning utilities for XLM-RoBERTa on XNLI.

A single :func:`build_trainer` entry point consumes an
:class:`~src.config.ExperimentConfig` and produces a configured
``transformers.Trainer``, eliminating the need to duplicate boilerplate per
experiment.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
import evaluate

from .config import MODEL_NAME, SEED, ExperimentConfig

# Reproducibility

def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for deterministic runs.

    Reference: https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Metrics

_accuracy_metric = None


def _get_accuracy_metric():
    global _accuracy_metric
    if _accuracy_metric is None:
        _accuracy_metric = evaluate.load("accuracy")
    return _accuracy_metric


def compute_metrics(eval_pred):
    """Classification accuracy computed on ``eval_pred.logits`` vs ``labels``."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return _get_accuracy_metric().compute(predictions=predictions, references=labels)

# Tokenizer

def load_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)

# Trainer factory

def build_trainer(
    experiment: ExperimentConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_root: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    model_name: str = MODEL_NAME,
    num_labels: int = 3,
) -> Trainer:
    """Instantiate a ``Trainer`` for a given experiment configuration.

    Sets the seed, loads a fresh classification head on the base model, and
    wires up the training arguments declared in :mod:`src.config`.
    """
    set_seed(SEED)

    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    output_dir = os.path.join(output_root, experiment.name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=experiment.batch_size,
        per_device_eval_batch_size=experiment.batch_size,
        num_train_epochs=experiment.num_train_epochs,
        learning_rate=experiment.learning_rate,
        weight_decay=experiment.weight_decay,
        warmup_ratio=experiment.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=experiment.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="tensorboard",
        seed=SEED,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
