"""
Central configuration for the XNLI cross-lingual fine-tuning experiments.

All hyperparameters for the four reported experiments live here so they can be
imported from scripts and reproduced deterministically.
"""

from dataclasses import dataclass, field
from typing import Dict


MODEL_NAME = "xlm-roberta-base"
SEED = 42
LABELS = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
SOURCE_LANG = "en"
TARGET_LANG = "tr"


@dataclass
class ExperimentConfig:
    """Hyperparameters for a single fine-tuning run."""

    name: str
    train_size_per_lang: int  # samples per language (total = 2 * this)
    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float = 0.1
    batch_size: int = 20
    logging_steps: int = 250
    notes: str = ""


EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "exp1_baseline": ExperimentConfig(
        name="Exp1_Baseline",
        train_size_per_lang=2500,
        num_train_epochs=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        notes="Standard fine-tuning; serves as reference point.",
    ),
    "exp2_regularised": ExperimentConfig(
        name="Exp2_Reg",
        train_size_per_lang=2500,
        num_train_epochs=4,
        learning_rate=2e-5,
        weight_decay=0.07,
        notes="Higher weight decay to counter the overfitting seen in Exp 1.",
    ),
    "exp3_low_lr": ExperimentConfig(
        name="Exp3_LowLR",
        train_size_per_lang=2500,
        num_train_epochs=6,
        learning_rate=1e-5,
        weight_decay=0.01,
        notes="Lower LR, more epochs - attempts a 'patient' accuracy peak.",
    ),
    "exp4_bigdata": ExperimentConfig(
        name="Exp4_BigData",
        train_size_per_lang=5000,
        num_train_epochs=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=500,
        notes="Doubles training data. Best-performing configuration.",
    ),
}

BEST_EXPERIMENT = "exp4_bigdata"
