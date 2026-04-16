"""
Hidden-state extraction for the TensorBoard Embedding Projector.

Extracts the start-token representation (``<s>`` at position 0 for RoBERTa-based
models) from selected encoder layers and writes TSV files compatible with the
Embedding Projector.

The original tutorial code looked up the ``[CLS]`` token ID used by BERT
tokenizers. XLM-RoBERTa uses ``<s>`` which is always at position 0 after
tokenization, so we index positionally instead.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import LABELS


def extract_layer_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    premises: List[str],
    hypotheses: List[str],
    labels: List[int],
    output_dir: str,
    layers_of_interest: Optional[Iterable[int]] = None,
    max_examples: int = 500,
    preview_chars: int = 40,
) -> List[str]:
    """Dump per-layer embeddings as TensorBoard projector logs.

    Parameters
    ----------
    model : PreTrainedModel
        A fine-tuned sequence-classification model.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer paired with ``model``.
    premises, hypotheses, labels
        Parallel lists of inputs and integer labels.
    output_dir : str
        Directory in which one subfolder per layer will be created.
    layers_of_interest : iterable of int, optional
        Which hidden-state indices to dump. Defaults to input (0), middle (6),
        and last encoder layer.
    max_examples : int
        Cap on the number of examples written per layer.
    preview_chars : int
        Length of the premise preview stored as embedding metadata.

    Returns
    -------
    list[str]
        Paths of the TensorBoard event directories that were written.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    premises = premises[:max_examples]
    hypotheses = hypotheses[:max_examples]
    labels = labels[:max_examples]

    model_inputs = tokenizer(
        premises,
        hypotheses,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True)

    hidden_states = outputs["hidden_states"]
    total_layers = len(hidden_states)

    if layers_of_interest is None:
        layers_of_interest = [0, total_layers // 2, total_layers - 1]
    layers_of_interest = set(layers_of_interest)

    os.makedirs(output_dir, exist_ok=True)
    written_paths: List[str] = []

    for layer_idx in range(total_layers):
        if layer_idx not in layers_of_interest:
            continue

        layer_name = f"Layer_{layer_idx}"
        if layer_idx == total_layers - 1:
            layer_name += "_Final"

        layer_path = os.path.join(output_dir, layer_name)
        os.makedirs(layer_path, exist_ok=True)

        # Index [0] is the start token <s> for RoBERTa tokenizers.
        tensors = [hidden_states[layer_idx][i][0].cpu() for i in range(len(premises))]
        metadata = [
            [premises[i][:preview_chars] + "...", LABELS[labels[i]]]
            for i in range(len(premises))
        ]

        writer = SummaryWriter(layer_path)
        writer.add_embedding(
            torch.stack(tensors),
            metadata=metadata,
            metadata_header=["Text", "Label"],
        )
        writer.close()
        print(f"  saved: {layer_name}")
        written_paths.append(layer_path)

    return written_paths
