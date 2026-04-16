"""
Extract hidden-state embeddings from a trained checkpoint for the
TensorBoard Embedding Projector.

The script reloads a checkpoint produced by ``evaluate_best.py`` /
``run_experiments.py`` and writes one event directory per selected layer into
``--output-dir``. Upload the resulting TSV files to
https://projector.tensorflow.org to reproduce the PCA/t-SNE views in the
report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForSequenceClassification

from src.data import build_mixed_validation, load_xnli_splits
from src.embeddings import extract_layer_embeddings
from src.train import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a fine-tuned model directory (contains config.json + weights).",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/embeddings",
        help="Where to write per-layer event directories.",
    )
    parser.add_argument(
        "--max-examples", type=int, default=500, help="Examples to embed."
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 6, 12],
        help="Hidden-state indices to dump (0 = input embeddings, 12 = final).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = load_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)

    xnli_en, xnli_tr = load_xnli_splits()
    val_multi = build_mixed_validation(xnli_en, xnli_tr)

    premises = val_multi["premise"][: args.max_examples]
    hypotheses = val_multi["hypothesis"][: args.max_examples]
    labels = val_multi["label"][: args.max_examples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_layer_embeddings(
        model=model,
        tokenizer=tokenizer,
        premises=premises,
        hypotheses=hypotheses,
        labels=labels,
        output_dir=str(output_dir),
        layers_of_interest=args.layers,
        max_examples=args.max_examples,
    )

    print(f"\nDone. Event logs written under {output_dir.resolve()}")
    print("Launch TensorBoard with:")
    print(f"  tensorboard --logdir {output_dir}")


if __name__ == "__main__":
    main()
