"""Multi-objective front metrics: Pareto front, hypervolume, IGD.

Domain-independent. Convention, pinned: every objective is minimized — pass
error rate (not accuracy) next to a cost such as parameter count or MACs.
"""

from collections.abc import Sequence

import numpy as np

from airas_eval.exceptions import UndefinedMetric


def _points(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        raise ValueError(
            f"{name} must have shape (n_points, n_objectives >= 2), got {arr.shape}"
        )
    return arr


def pareto_front_mask(points: Sequence[Sequence[float]]) -> list[bool]:
    """Non-dominated mask under minimize-all-objectives.

    A point is dominated if some other point is <= in every objective and <
    in at least one. Exact duplicates of a front point are all on the front.
    """
    arr = _points(points, "points")
    mask: list[bool] = []
    for candidate in arr:
        dominated = bool(
            np.any(np.all(arr <= candidate, axis=1) & np.any(arr < candidate, axis=1))
        )
        mask.append(not dominated)
    return mask


def pareto_front(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """Non-dominated points, deduplicated and sorted by the first objective."""
    arr = _points(points, "points")
    front = arr[np.array(pareto_front_mask(points))]
    front = np.unique(front, axis=0)
    return [[float(v) for v in row] for row in front]


def hypervolume_2d(
    points: Sequence[Sequence[float]],
    reference_point: Sequence[float],
) -> float:
    """Exact hypervolume for two minimized objectives.

    The area dominated by the non-dominated points and bounded by
    ``reference_point`` (the worst acceptable value per objective, fixed in
    the experimental design, not chosen after seeing results). Points that
    do not strictly dominate the reference point contribute nothing.
    """
    arr = _points(points, "points")
    if arr.shape[1] != 2:
        raise ValueError("hypervolume_2d requires exactly 2 objectives")
    ref = np.asarray(reference_point, dtype=float)
    if ref.shape != (2,):
        raise ValueError("reference_point must have exactly 2 objectives")
    contributing = arr[np.all(arr < ref, axis=1)]
    if len(contributing) == 0:
        return 0.0
    # np.unique sorts ascending by objective 1; on a non-dominated 2D front
    # objective 2 is then strictly decreasing, so a single sweep is exact.
    front = np.unique(contributing[np.array(pareto_front_mask(contributing))], axis=0)
    right_edges = np.append(front[1:, 0], ref[0])
    return float(np.sum((right_edges - front[:, 0]) * (ref[1] - front[:, 1])))


def _check_same_objectives(a: np.ndarray, b: np.ndarray, b_name: str) -> None:
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"objective count mismatch: points have {a.shape[1]}, "
            f"{b_name} has {b.shape[1]}"
        )


def gd(
    points: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
) -> float:
    """Generational distance: mean Euclidean distance from each obtained point
    to the nearest reference-front point (convergence; IGD measures coverage).
    Scale-sensitive: normalize objectives to comparable ranges before calling.
    """
    obtained = _points(points, "points")
    front = _points(reference_front, "reference_front")
    _check_same_objectives(obtained, front, "reference_front")
    distances = np.linalg.norm(obtained[:, None, :] - front[None, :, :], axis=2)
    return float(distances.min(axis=1).mean())


def spacing(points: Sequence[Sequence[float]]) -> float:
    """Schott's spacing: standard deviation of nearest-neighbour distances
    among the non-dominated points (0 = perfectly even). Needs a front of at
    least two distinct points; scale-sensitive like the other distances.
    """
    front = np.asarray(pareto_front(points), dtype=float)
    if len(front) < 2:
        raise UndefinedMetric("spacing needs at least two distinct front points")
    distances = np.linalg.norm(front[:, None, :] - front[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.std(distances.min(axis=1)))


def igd(
    points: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
) -> float:
    """Inverted generational distance to a known reference front.

    Mean Euclidean distance from each reference-front point to the nearest
    obtained point. Scale-sensitive: normalize objectives to comparable
    ranges before calling.
    """
    obtained = _points(points, "points")
    front = _points(reference_front, "reference_front")
    _check_same_objectives(obtained, front, "reference_front")
    distances = np.linalg.norm(front[:, None, :] - obtained[None, :, :], axis=2)
    return float(distances.min(axis=1).mean())
