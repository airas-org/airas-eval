"""Structural-biology metrics.

Policy: metrics with a community-accepted reference implementation are wrapped,
not reimplemented — an in-house lDDT or DockQ would be exactly the
"self-implemented metric" failure mode this library exists to prevent.

- In-house (simple, unambiguous linear algebra): RMSD with Kabsch superposition.
- Wrapped (optional ``structure`` extra): TM-score via ``tmtools``,
  DockQ via the official ``DockQ`` package.
"""

from collections.abc import Sequence

import numpy as np


def _paired_coords(
    predicted: Sequence[Sequence[float]], reference: Sequence[Sequence[float]]
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(predicted, dtype=float)
    r = np.asarray(reference, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_atoms, 3)")
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {r.shape}")
    if len(p) == 0:
        raise ValueError("cannot compute a metric on zero atoms")
    return p, r


def rmsd(
    predicted: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    superpose: bool = True,
) -> float:
    """Root-mean-square deviation between paired atom coordinates (Angstrom).

    With ``superpose=True`` (default) the predicted coordinates are optimally
    superposed onto the reference by the Kabsch algorithm first.
    """
    p, r = _paired_coords(predicted, reference)
    if superpose:
        p = kabsch_superpose(p, r)
    return float(np.sqrt(np.mean(np.sum((p - r) ** 2, axis=1))))


def kabsch_superpose(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Optimal rigid-body superposition of ``mobile`` onto ``target`` (Kabsch)."""
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    mob_center = mobile - mobile.mean(axis=0)
    tgt_center = target - target.mean(axis=0)
    h = mob_center.T @ tgt_center
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = vt.T @ correction @ u.T
    return (rotation @ mob_center.T).T + target.mean(axis=0)


def tm_score(
    predicted: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    reference_sequence: str,
) -> float:
    """TM-score via ``tmtools`` (bindings to the official TM-align code).

    Requires the ``structure`` extra. Coordinates are C-alpha traces.
    """
    from tmtools import tm_align  # noqa: PLC0415 - optional dependency

    p, r = _paired_coords(predicted, reference)
    result = tm_align(p, r, reference_sequence, reference_sequence)
    return float(result.tm_norm_chain2)


def dockq(model_path: str, native_path: str) -> dict[str, float]:
    """DockQ via the official DockQ package (v2), on structure files.

    Requires the ``structure`` extra. Returns the best mapping's summary
    including ``DockQ``, ``iRMSD``, ``LRMSD`` and ``fnat`` per interface,
    aggregated as the official CLI does.
    """
    from DockQ.DockQ import (  # noqa: PLC0415 - optional dependency
        load_PDB,
        run_on_all_native_interfaces,
    )

    model = load_PDB(model_path)
    native = load_PDB(native_path)
    results, total = run_on_all_native_interfaces(model, native)
    if not results:
        raise ValueError("DockQ found no comparable interfaces")
    summary: dict[str, float] = {"DockQ": float(total) / len(results)}
    for interface, values in results.items():
        summary[f"DockQ_{'_'.join(interface)}"] = float(values["DockQ"])
    return summary
