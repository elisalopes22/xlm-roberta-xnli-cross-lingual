"""
Run one or more of the XNLI fine-tuning experiments defined in ``src.config``.

Examples
--------
Run all four experiments in sequence::

    python scripts/run_experiments.py --all

Run just the best-performing configuration::

    python scripts/run_experiments.py --experiment exp4_bigdata
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import EXPERIMENTS
from src.data import (
    build_mixed_train,
    build_mixed_validation,
    load_xnli_splits,
    tokenize_splits,
)
from src.evaluate import evaluate_per_language, pretty_print_results
from src.train import build_trainer, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS),
        help="Single experiment key to run.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run every experiment defined in config."
    )
    parser.add_argument(
        "--output-root",
        default="./results/checkpoints",
        help="Directory for checkpoints and TensorBoard logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all and not args.experiment:
        raise SystemExit("Use --all or --experiment <key>.")

    to_run = list(EXPERIMENTS) if args.all else [args.experiment]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Data loading happens once, up front.
    tokenizer = load_tokenizer()
    xnli_en, xnli_tr = load_xnli_splits()
    val_multi = build_mixed_validation(xnli_en, xnli_tr)

    tokenized_val = tokenize_splits(
        {"en": xnli_en["validation"], "tr": xnli_tr["validation"], "mixed": val_multi},
        tokenizer,
    )

    summary = {}

    for key in to_run:
        cfg = EXPERIMENTS[key]
        print(f"\n{'=' * 70}\nRunning {cfg.name}\n{'=' * 70}")
        print(f"  {cfg.notes}")

        train_ds = build_mixed_train(xnli_en, xnli_tr, cfg.train_size_per_lang)
        tokenized_train = tokenize_splits({"train": train_ds}, tokenizer)["train"]

        trainer = build_trainer(
            experiment=cfg,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val["mixed"],
            output_root=str(output_root),
            tokenizer=tokenizer,
        )
        trainer.train()

        results = evaluate_per_language(trainer, tokenized_val)
        pretty_print_results(results, title=f"{cfg.name} (validation)")
        summary[cfg.name] = results

    print("\n" + "=" * 70)
    print("Summary of validation accuracies")
    print("=" * 70)
    for name, res in summary.items():
        print(f"  {name}")
        for lang, acc in res.items():
            print(f"    {lang:<8s} {acc:.4f}")


if __name__ == "__main__":
    main()
