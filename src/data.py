"""
Data loading and tokenization for the XNLI cross-lingual experiments.

Builds a mixed EN+TR training set and a mixed validation set so the model is
monitored on both languages simultaneously during fine-tuning.
"""

from typing import Dict, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import SEED, SOURCE_LANG, TARGET_LANG


def load_xnli_splits() -> Tuple[Dict[str, Dataset], Dict[str, Dataset]]:
    """Load the full XNLI EN and TR splits.

    Returns
    -------
    (xnli_en, xnli_tr)
        Dataset dicts with ``train``, ``validation`` and ``test`` splits.
    """
    xnli_en = load_dataset("xnli", SOURCE_LANG)
    xnli_tr = load_dataset("xnli", TARGET_LANG)
    return xnli_en, xnli_tr


def build_mixed_train(
    xnli_en: Dict[str, Dataset],
    xnli_tr: Dict[str, Dataset],
    samples_per_lang: int,
    seed: int = SEED,
) -> Dataset:
    """Build a balanced EN+TR training subset.

    ``samples_per_lang`` examples are drawn from each language, so the resulting
    dataset has ``2 * samples_per_lang`` rows.
    """
    train_en = xnli_en["train"].shuffle(seed=seed).select(range(samples_per_lang))
    train_tr = xnli_tr["train"].shuffle(seed=seed).select(range(samples_per_lang))
    return concatenate_datasets([train_en, train_tr])


def build_mixed_validation(
    xnli_en: Dict[str, Dataset], xnli_tr: Dict[str, Dataset]
) -> Dataset:
    """Concatenate the EN and TR validation splits for mixed-language monitoring."""
    return concatenate_datasets([xnli_en["validation"], xnli_tr["validation"]])


def make_tokenize_fn(tokenizer: PreTrainedTokenizerBase):
    """Return a tokenization function suited to XNLI's premise/hypothesis pairs."""

    def _tokenize(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
        )

    return _tokenize


def tokenize_splits(
    datasets: Dict[str, Dataset], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, Dataset]:
    """Apply tokenization across every split in a dict of datasets."""
    tok_fn = make_tokenize_fn(tokenizer)
    return {name: ds.map(tok_fn, batched=True) for name, ds in datasets.items()}
