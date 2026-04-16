"""
Train the best configuration (Experiment 4) and evaluate on the XNLI test set.

The test split is held out from training and validation. This script reports
accuracy separately for English and Turkish to surface the cross-lingual
transfer gap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import BEST_EXPERIMENT, EXPERIMENTS
from src.data import (
    build_mixed_train,
    build_mixed_validation,
    load_xnli_splits,
    tokenize_splits,
)
from src.error_analysis import collect_predictions, summarise
from src.evaluate import evaluate_per_language, pretty_print_results
from src.train import build_trainer, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default=BEST_EXPERIMENT,
        choices=list(EXPERIMENTS),
        help="Which experiment configuration to train and evaluate.",
    )
    parser.add_argument(
        "--output-root",
        default="./results/checkpoints",
        help="Directory for checkpoints and TensorBoard logs.",
    )
    parser.add_argument(
        "--skip-error-analysis",
        action="store_true",
        help="Skip the qualitative error dump.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EXPERIMENTS[args.experiment]

    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    xnli_en, xnli_tr = load_xnli_splits()

    val_multi = build_mixed_validation(xnli_en, xnli_tr)
    train_ds = build_mixed_train(xnli_en, xnli_tr, cfg.train_size_per_lang)

    tokenized = tokenize_splits(
        {
            "train": train_ds,
            "val_en": xnli_en["validation"],
            "val_tr": xnli_tr["validation"],
            "val_mixed": val_multi,
            "test_en": xnli_en["test"],
            "test_tr": xnli_tr["test"],
        },
        tokenizer,
    )

    trainer = build_trainer(
        experiment=cfg,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val_mixed"],
        output_root=args.output_root,
        tokenizer=tokenizer,
    )
    trainer.train()

    val_results = evaluate_per_language(
        trainer, {"EN": tokenized["val_en"], "TR": tokenized["val_tr"]}
    )
    pretty_print_results(val_results, title=f"{cfg.name} - Validation")

    test_results = evaluate_per_language(
        trainer, {"EN": tokenized["test_en"], "TR": tokenized["test_tr"]}
    )
    pretty_print_results(test_results, title=f"{cfg.name} - Test set (held out)")

    gap = test_results["EN"] - test_results["TR"]
    print(f"\n  EN -> TR transfer gap on test: {gap:+.4f}")

    if not args.skip_error_analysis:
        print("\n" + "=" * 70)
        print("Qualitative error analysis")
        print("=" * 70)
        for name in ("val_tr", "test_tr", "test_en"):
            records = collect_predictions(trainer, tokenized[name])
            summarise(records, dataset_name=name)


if __name__ == "__main__":
    main()
