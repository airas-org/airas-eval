import numpy as np
import pytest

from airas_eval.metrics import structure as st

rng = np.random.default_rng(11)
COORDS = rng.normal(0, 5, size=(30, 3))


def _random_rotation() -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_rmsd_identical_is_zero():
    assert st.rmsd(COORDS, COORDS) == pytest.approx(0.0, abs=1e-9)


def test_rmsd_invariant_under_rigid_motion_with_superposition():
    moved = COORDS @ _random_rotation().T + np.array([10.0, -3.0, 7.0])
    assert st.rmsd(moved, COORDS, superpose=True) == pytest.approx(0.0, abs=1e-8)


def test_rmsd_without_superposition_sees_the_shift():
    shifted = COORDS + np.array([1.0, 0.0, 0.0])
    assert st.rmsd(shifted, COORDS, superpose=False) == pytest.approx(1.0)


def test_rmsd_with_noise_matches_manual():
    noisy = COORDS + rng.normal(0, 0.1, size=COORDS.shape)
    manual = float(np.sqrt(np.mean(np.sum((noisy - COORDS) ** 2, axis=1))))
    assert st.rmsd(noisy, COORDS, superpose=False) == pytest.approx(manual)
    # Superposition can only reduce (or keep) the RMSD.
    assert st.rmsd(noisy, COORDS, superpose=True) <= manual + 1e-12


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        st.rmsd(COORDS, COORDS[:-1])
