# Experiment Results

All runs fine-tune `xlm-roberta-base` on a balanced EN+TR training subset of
XNLI, with the multilingual validation split used for model selection. The
seed is fixed (`42`) across Python, NumPy and PyTorch.

## Validation accuracy (mixed EN+TR)

| Experiment | Train size | Epochs | LR    | Weight decay | Val accuracy | Val loss | Observation                         |
| ---------- | ---------- | ------ | ----- | ------------ | ------------ | -------- | ----------------------------------- |
| Exp 1      | 5 000      | 4      | 2e-5  | 0.01         | 64.18%       | 0.8916   | Overfitting after epoch 3           |
| Exp 2      | 5 000      | 4      | 2e-5  | 0.07         | 60.50%       | 0.9710   | Over-regularised / underfitting     |
| Exp 3      | 5 000      | 6      | 1e-5  | 0.01         | 63.51%       | 0.9447   | Severe overfitting after epoch 4    |
| **Exp 4**  | **10 000** | **4**  | **2e-5** | **0.01** | **69.92%** | **0.8803** | **Best configuration**           |

## Final test-set accuracy (Experiment 4)

Evaluated on the official XNLI test split (never seen during training or model selection).

| Language          | Test accuracy |
| ----------------- | ------------- |
| English (source)  | 72.36%        |
| Turkish (target)  | 66.21%        |
| **Transfer gap**  | **6.15 pp**   |

## Notes

- The main driver of the improvement in Exp 4 was training data size (5k → 10k examples), not hyperparameters.
- The 6.15 pp EN→TR gap is consistent with the expected degradation for a mid-resource, morphologically distant target language fine-tuned with limited data.
- Qualitative error patterns are discussed in the report (`report/final_report.pdf`, §5).

## Reproducibility note

A second run on an updated Colab environment (different `transformers`/CUDA
versions, same seed) produced results that match the reported ones within
noise, confirming the experimental pattern rather than the exact numbers:

| Metric          | Report | Reproduction | Δ        |
| --------------- | ------ | ------------ | -------- |
| Exp 4 (mixed)   | 69.92% | 67.87%       | -2.05 pp |
| Test EN         | 72.36% | 71.78%       | -0.58 pp |
| Test TR         | 66.21% | 65.55%       | -0.66 pp |
| EN→TR gap       | 6.15 pp| 6.23 pp      | +0.08 pp |

Exp 4 remained the best configuration and the cross-lingual gap stayed
essentially unchanged. Small absolute differences are expected.
