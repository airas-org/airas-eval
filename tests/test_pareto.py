import numpy as np
import pytest

from airas_eval.exceptions import UndefinedMetric
from airas_eval.metrics import pareto


def test_pareto_front_mask_and_front():
    points = [[1.0, 2.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
    assert pareto.pareto_front_mask(points) == [True, True, False, True]
    assert pareto.pareto_front(points) == [[1.0, 2.0], [2.0, 1.0]]


def test_hypervolume_single_point():
    assert pareto.hypervolume_2d([[1.0, 1.0]], [3.0, 3.0]) == pytest.approx(4.0)


def test_hypervolume_hand_computed():
    points = [[1.0, 2.0], [2.0, 1.0]]
    # (2-1)*(3-2) + (3-2)*(3-1) = 1 + 2
    assert pareto.hypervolume_2d(points, [3.0, 3.0]) == pytest.approx(3.0)


def test_hypervolume_ignores_dominated_and_out_of_reference():
    base = [[1.0, 2.0], [2.0, 1.0]]
    hv = pareto.hypervolume_2d(base, [3.0, 3.0])
    assert pareto.hypervolume_2d(base + [[2.5, 2.5]], [3.0, 3.0]) == pytest.approx(hv)
    assert pareto.hypervolume_2d(base + [[4.0, 0.5]], [3.0, 3.0]) == pytest.approx(hv)
    assert pareto.hypervolume_2d([[4.0, 4.0]], [3.0, 3.0]) == 0.0


def test_hypervolume_monotone_under_nondominated_addition():
    base = [[1.0, 2.0], [2.0, 1.0]]
    improved = base + [[0.5, 2.5]]
    assert pareto.hypervolume_2d(improved, [3.0, 3.0]) > pareto.hypervolume_2d(
        base, [3.0, 3.0]
    )


def test_hypervolume_parity_with_grid_integration():
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 1.0, size=(20, 2)).tolist()
    exact = pareto.hypervolume_2d(points, [1.0, 1.0])
    n = 800
    xs = (np.arange(n) + 0.5) / n
    grid = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    arr = np.asarray(points)
    dominated = np.zeros(len(grid), dtype=bool)
    for p in arr:
        dominated |= np.all(grid >= p, axis=1)
    assert exact == pytest.approx(dominated.mean(), abs=5e-3)


def test_igd():
    front = [[0.0, 1.0], [1.0, 0.0]]
    assert pareto.igd(front, front) == 0.0
    assert pareto.igd([[0.0, 2.0], [2.0, 0.0]], front) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="objective count mismatch"):
        pareto.igd([[0.0, 1.0, 2.0]], front)


def test_points_shape_validation():
    with pytest.raises(ValueError, match="shape"):
        pareto.pareto_front_mask([[1.0], [2.0]])
    with pytest.raises(ValueError, match="exactly 2"):
        pareto.hypervolume_2d([[1.0, 2.0, 3.0]], [1.0, 1.0, 1.0])


def test_gd_mirrors_igd():
    front = [[0.0, 1.0], [1.0, 0.0]]
    assert pareto.gd(front, front) == 0.0
    # one obtained point far away: GD sees it, IGD (per-reference nearest) does
    # not fully — the two measure convergence vs coverage
    obtained = [[0.0, 1.0], [1.0, 0.0], [3.0, 3.0]]
    assert pareto.gd(obtained, front) == pytest.approx(np.sqrt(13.0) / 3)
    assert pareto.igd(obtained, front) == 0.0
    with pytest.raises(ValueError, match="objective count mismatch"):
        pareto.gd([[0.0, 1.0, 2.0]], front)


def test_spacing():
    even = [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]
    assert pareto.spacing(even) == pytest.approx(0.0)
    uneven = [[0.0, 3.0], [0.1, 2.9], [3.0, 0.0]]
    assert pareto.spacing(uneven) > 0.0
    # dominated points are ignored; duplicates collapse
    assert pareto.spacing(even + [[5.0, 5.0], [0.0, 3.0]]) == pytest.approx(0.0)
    with pytest.raises(UndefinedMetric):
        pareto.spacing([[1.0, 1.0], [1.0, 1.0]])
