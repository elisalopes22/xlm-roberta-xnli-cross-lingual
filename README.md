# Cross-Lingual Natural Language Inference with XLM-RoBERTa

Fine-tuning and linguistic error analysis of
[XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base) on the
English and Turkish splits of [XNLI](https://huggingface.co/datasets/xnli),
with a focus on how the model handles Turkish agglutinative morphology and
cross-lingual transfer.

Final project for *Introduction to Computational Linguistics*, University of
Vienna, Winter 2026.

## Key findings

- The best configuration reaches **72.36%** on the English test set and
  **66.21%** on the Turkish test set after fine-tuning on only 10 000 mixed
  EN+TR examples, giving a **6.15 pp cross-lingual transfer gap**.
- Increasing training data (5k → 10k) yielded a larger accuracy gain than any
  of the regularisation or learning-rate strategies that were tried.
- The dominant Turkish error mode is a systematic bias toward predicting
  `Contradiction` when lexical overlap is weak. Qualitative inspection
  (consulted with a native speaker) ties several of these errors to
  morphological fragmentation under subword tokenization, the directional
  ambiguity of verbs such as *gitmek*, and to a few gold labels that are
  themselves pragmatically debatable once you look at the Turkish sentence
  carefully.
- Layer-wise PCA of the `<s>` embeddings shows the expected progression:
  input embeddings are collapsed, mid-layer representations are diffuse, and
  only the final layer organises inputs into clusters aligned with the NLI
  labels (across both languages).

A full discussion, including specific error cases with morphological glosses,
is in [`report/final_report.pdf`](report/final_report.pdf).

## Repository layout

```
xlm-roberta-xnli-cross-lingual/
├── src/                         # Reusable modules
│   ├── config.py                # Experiment hyperparameters
│   ├── data.py                  # XNLI loading + tokenization
│   ├── train.py                 # Trainer factory + seeding
│   ├── evaluate.py              # Per-language accuracy helpers
│   ├── embeddings.py            # Hidden-state extraction for TB projector
│   └── error_analysis.py        # Qualitative prediction inspection
├── scripts/
│   ├── run_experiments.py       # Run one or all four experiments
│   ├── evaluate_best.py         # Train best config and test on held-out split
│   └── extract_embeddings.py    # Dump per-layer embeddings from a checkpoint
├── notebooks/
│   └── 01_full_pipeline.ipynb   # End-to-end walkthrough (Colab-friendly)
├── results/
│   ├── experiments_summary.md   # Results table
│   └── figures/                 # PCA visualizations of hidden states
├── report/
│   └── final_report.pdf         # Full written report with linguistic analysis
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

The experiments in the report were run on a Google Colab L4.

```bash
git clone https://github.com/elisalopes22/xlm-roberta-xnli-cross-lingual.git
cd xlm-roberta-xnli-cross-lingual

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Reproducing the experiments

Every run is seeded (`SEED=42` across Python, NumPy and PyTorch), so the
numbers in the report should be reproducible within floating-point tolerance
on comparable hardware.

Run a single experiment:

```bash
python scripts/run_experiments.py --experiment exp4_bigdata
```

Run all four experiments in sequence:

```bash
python scripts/run_experiments.py --all
```

Train the best configuration and evaluate on the held-out test set, with a
qualitative error dump per language:

```bash
python scripts/evaluate_best.py
```

Extract hidden-state embeddings for the TensorBoard Embedding Projector from
a trained checkpoint:

```bash
python scripts/extract_embeddings.py \
    --checkpoint results/checkpoints/Exp4_BigData/checkpoint-<step> \
    --output-dir results/embeddings \
    --layers 0 6 12

tensorboard --logdir results/embeddings
```

To reproduce the PCA views in the report, upload the generated TSV pairs to
[projector.tensorflow.org](https://projector.tensorflow.org/).

## Experimental design

Four configurations were compared under a fixed training/validation
protocol. Validation uses the full concatenated EN+TR XNLI validation split
so that generalisation is monitored jointly across both languages at each
epoch.

| Experiment | Train size (balanced EN+TR) | Epochs | LR    | Weight decay | Motivation                     |
| ---------- | --------------------------- | ------ | ----- | ------------ | ------------------------------ |
| Exp 1      | 5 000                       | 4      | 2e-5  | 0.01         | Baseline                       |
| Exp 2      | 5 000                       | 4      | 2e-5  | 0.07         | Address Exp 1 overfitting      |
| Exp 3      | 5 000                       | 6      | 1e-5  | 0.01         | Slower training, more epochs   |
| **Exp 4** (best) | **10 000**            | **4**  | **2e-5** | **0.01** | **Double the training data** |

See [`results/experiments_summary.md`](results/experiments_summary.md) for
the accuracy/loss table.

## Implementation notes

A few choices depart from the course tutorials and are worth flagging:

- **Mixed-language validation.** `build_mixed_validation` concatenates the EN
  and TR validation splits so `eval_accuracy` reflects cross-lingual
  generalisation at every epoch, not just English performance.
- **Start-token indexing.** XLM-RoBERTa uses `<s>` at position 0; the course
  tutorial's `[CLS]` token lookup is BERT-specific. `src/embeddings.py`
  indexes the hidden states positionally with `[0]` instead.
- **Deterministic runs.** `src.train.set_seed` seeds Python, NumPy, and
  PyTorch (CPU + CUDA) so accuracy improvements across experiments can be
  attributed to data or hyperparameter changes rather than random
  initialization.

## Limitations

- The training budget is modest (≤10k examples, 4 epochs). Absolute accuracy
  is lower than what full XNLI fine-tuning or translate-train approaches
  achieve. The project is scoped as an analysis of cross-lingual transfer
  behavior, not as an attempt at SOTA numbers.
- XNLI Turkish is produced by machine translation from English, and as the
  report discusses, some Turkish sentences carry translation artefacts
  (e.g. masculine default for gender-neutral English pronouns), which
  contaminates a subset of the test cases.

## Citation

- Conneau et al. (2020), *Unsupervised Cross-lingual Representation Learning
  at Scale* ([XLM-R](https://arxiv.org/abs/1911.02116)).
- Conneau et al. (2018), *XNLI: Evaluating Cross-lingual Sentence
  Representations* ([XNLI](https://arxiv.org/abs/1809.05053)).
