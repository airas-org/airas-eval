# airas-eval

The trusted evaluation layer for [AIRAS](https://github.com/airas-org/airas):
agents pass a task type and raw predictions, and get the full standard metric
suite back. Agents never implement evaluation scripts and never choose which
metrics (or which variants) get reported.

## The model

```python
from airas_eval import evaluate

report = evaluate(
    "classification",
    {
        "predicted_labels": y_pred,
        "reference_labels": y_true,
        "probabilities": probs,  # optional
    },
)
report.metrics  # every standard metric at pinned variants
report.skipped  # metrics that don't apply, each with a reason
report.provenance  # suite signature, resolved package versions, input SHA-256
```

Or as a process boundary, for trusted scoring jobs:

```bash
airas-eval score classification --inputs inputs.json --output evaluation.json
airas-eval list   # what suites exist and what inputs they take
```

Design rules, in order of importance:

1. **No metric choice.** A suite computes *all* standard metrics for the task
   type. F1 is reported as macro *and* micro *and* weighted. Reporting
   everything is what removes the cherry-picking degree of freedom.
2. **No agent-written metric code.** Computation is delegated to the
   community-canonical implementation for each metric — scikit-learn / scipy
   for the classic ML metrics, `sacrebleu` for BLEU/chrF, Google's
   `rouge-score` for ROUGE, official `DockQ` v2, `tmtools` for TM-score,
   `fvcore` for MACs. airas-eval implements a metric itself only when no
   canonical pip implementation exists (ECE, SQuAD-style EM/F1, ranking@k,
   Kabsch RMSD), and pins the variant explicitly.
3. **Nothing disappears silently.** A metric that cannot be computed on the
   given inputs (missing probabilities, not binary, missing optional
   dependency, mathematically undefined) appears under `skipped` with a
   reason. Unknown task types, unknown input keys, and missing required inputs
   raise.
4. **Every report carries provenance.** The suite signature (task type +
   pinned variants), the resolved versions of the packages that actually
   computed the numbers, and a SHA-256 of the inputs. Pin airas-eval by commit
   in the scoring job and a reported score is reproducible as
   `(inputs, airas-eval@version) -> metrics`.

What this library does *not* claim: it cannot force agents to call it, and it
does not verify that the predictions themselves are genuine (that they came
from the claimed model run, or cover the full test split). Those guarantees
belong to the surrounding infrastructure — a scoring job the agent cannot
edit, and input-provenance verification — which consume this library.

## Install

```bash
uv add airas-eval                       # core: numpy, scikit-learn, scipy
uv add "airas-eval[nlp]"                # + sacrebleu, rouge-score
uv add "airas-eval[structure]"          # + DockQ, tmtools
uv add "airas-eval[complexity]"         # + torch, fvcore (params / MACs)
```

## Suites

| Task type | Required inputs | Optional | Metrics |
|---|---|---|---|
| `classification` | predicted_labels, reference_labels | probabilities | accuracy, error rate, P/R/F1 (macro+micro+weighted), balanced accuracy, MCC, Cohen's kappa; with probabilities: log loss, ECE, top-5; binary only: AUROC, average precision, Brier |
| `regression` | predicted_values, reference_values | — | MSE, RMSE, MAE, MAPE (strict), sMAPE, R², explained variance, Pearson, Spearman, Kendall tau-b |
| `clustering` | predicted_labels, reference_labels | — | ARI, NMI, AMI, V-measure |
| `retrieval` | ranked_lists, relevant_sets | relevances | P@1/5/10, R@5/10, MRR, MAP, nDCG@10 |
| `text_qa` | predicted_texts, reference_texts | — | exact match, token F1 (SQuAD normalization) |
| `text_generation` | predicted_texts, reference_texts | — | BLEU, chrF (sacrebleu), ROUGE-L (rouge-score) |
| `segmentation` | predicted_mask, reference_mask | — | pixel accuracy, mIoU, Dice |
| `structure_comparison` | predicted_coords, reference_coords | reference_sequence | RMSD (Kabsch), RMSD w/o superposition, TM-score (tmtools) |

The underlying metric functions (`airas_eval.metrics.*`) remain importable for
trusted-side use, along with statistical helpers (`stats.mean_std`,
`stats.bootstrap_ci`, `stats.paired_permutation_test`) and model-complexity
utilities (`complexity.parameter_count`, `complexity.macs`).

## Roadmap

- lDDT / lDDT-PLI as an [OpenStructure](https://openstructure.org/) wrapper
  (reference implementation; conda/container only, hence not yet an extra).
- COCO mAP as a `pycocotools` wrapper; DockQ suite for structure files.
- SSIM (window/data-range conventions), CRPS (`properscoring`), MASE.
- NAS suite: architecture-level scoring (params/MACs from a spec) and
  benchmark-oracle protocols.

Out of scope by design: pLDDT and other self-reported model confidences (they
compare against no reference), latency/memory benchmarking (hardware-dependent),
and verifying the authenticity of prediction inputs (an infrastructure concern,
not a metric concern).

## Development

```bash
uv sync
uv run ruff check .
uv run mypy src
uv run pytest
```

Core metric outputs are tested for parity against scikit-learn / scipy,
including tie handling; the suite layer is tested for its contract (fail-closed
inputs, explicit skips, provenance determinism).

## License

MIT
