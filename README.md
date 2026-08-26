# airas-eval

Standard evaluation-metric implementations for [AIRAS](https://github.com/airas-org/airas):
a trusted, versioned scoring layer that computes metrics from raw predictions,
so that agent-generated experiments never score themselves.

## Why this exists

Autonomous research agents can (and do) game evaluations: hand-rolled metric
implementations, favorable variants chosen silently, and self-reported numbers
that nothing recomputes. `airas-eval` is the countermeasure:

- **Metrics are standard or wrapped, never invented.** Metrics with an
  unambiguous mathematical definition are implemented here as pure functions
  `(predictions, references) -> score` and verified in tests against
  scikit-learn / scipy as parity oracles. Metrics whose community-accepted
  reference implementation is a specific package (BLEU → `sacrebleu`,
  DockQ → official `DockQ`, TM-score → `tmtools`) are thin wrappers — a
  self-implemented DockQ would be exactly the failure mode this library exists
  to prevent.
- **Variants are explicit.** Where a metric has ambiguous variants (F1
  averaging, ECE binning, sMAPE denominators, MACs vs FLOPs), the variant is an
  explicit argument or is stated in the function's contract, never silently
  chosen.
- **Versioned scoring.** Pin this package by commit or release in the trusted
  scoring workflow; a reported score is then reproducible as
  `(raw predictions, airas-eval@version) -> metric`.

## Install

```bash
uv add airas-eval                       # core: numpy only
uv add "airas-eval[nlp]"                # + sacrebleu, rouge-score
uv add "airas-eval[structure]"          # + DockQ, tmtools
uv add "airas-eval[complexity]"         # + torch, fvcore (params / MACs)
```

## Usage

```python
from airas_eval.metrics import classification, stats

acc = classification.accuracy(predicted_labels, reference_labels)
f1 = classification.f1(predicted_labels, reference_labels, average="macro")
ci = stats.bootstrap_ci(classification.accuracy, predicted_labels, reference_labels)
```

Inspect what is registered:

```bash
airas-eval list                # all task types
airas-eval list classification
```

## Coverage

| Task type | Metrics | Origin |
|---|---|---|
| classification | accuracy, error rate, top-k, precision/recall/F1 (micro/macro/weighted), balanced accuracy, MCC, Cohen's kappa, AUROC, average precision, log loss, Brier, ECE | core |
| regression | MSE, RMSE, MAE, MAPE, sMAPE, R², explained variance, Pearson, Spearman, Kendall tau-b | core |
| ranking | precision@k, recall@k, hit rate@k, MRR, MAP, nDCG@k | core |
| clustering | ARI, NMI, AMI, V-measure | core |
| vision | pixel accuracy, IoU, Dice, mIoU, PSNR | core |
| nlp | exact match, token F1 (SQuAD-style) | core |
| nlp | BLEU, chrF (sacrebleu), ROUGE-L (rouge-score) | wrapped |
| structure | RMSD with Kabsch superposition | core |
| structure | TM-score (tmtools), DockQ (official DockQ v2) | wrapped |
| complexity | parameter count (torch), MACs with explicit counter metadata (fvcore) | wrapped |
| stats | mean±std over seeds, bootstrap CI for any metric, paired permutation test | core |

Every core metric is tested for parity against scikit-learn / scipy where an
oracle exists, including tie handling.

## Roadmap

- lDDT / lDDT-PLI as an [OpenStructure](https://openstructure.org/) wrapper
  (reference implementation; conda/container only, hence not yet an extra).
- COCO mAP as a `pycocotools` wrapper (protocol details make reimplementation
  a known source of errors).
- SSIM (window/data-range conventions), CRPS (`properscoring`), MASE.
- Score provenance signatures (sacrebleu-style config strings) on every result.
- Model-dependent scores (FID via `clean-fid`, LPIPS, BERTScore) as clearly
  separated optional wrappers — never mixed into the pure core.

Out of scope by design: pLDDT and other self-reported model confidences (they
compare against no reference and are not evaluation metrics), latency and
memory benchmarking (hardware-dependent; belongs in a benchmarking harness).

## Development

```bash
uv sync
uv run ruff check .
uv run mypy src
uv run pytest
```

## License

MIT
