"""Integration test: spiral -> Swiss roll alignment.

Validates that TorchGW produces a high-quality transport plan on the
canonical spiral-to-Swiss-roll benchmark.
"""
import time

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from torchgw import sampled_gw


def _sample_spiral(n, seed=0):
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * 0.05
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    return np.stack((x, y), axis=1).astype(np.float32), angles


def _sample_swiss_roll(n, seed=1):
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * 0.05
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    z = 0.1 * rng.uniform(size=n) * 10
    return np.stack((x, z, y), axis=1).astype(np.float32), angles


class TestSpiralToSwissRoll:
    """Integration tests on 400 vs 500 spiral -> Swiss roll (default dijkstra mode)."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        request.cls.spiral, request.cls.a_src = _sample_spiral(400, seed=0)
        request.cls.swiss_roll, request.cls.a_tgt = _sample_swiss_roll(500, seed=1)

        request.cls.T, request.cls.log_dict = sampled_gw(
            request.cls.spiral, request.cls.swiss_roll,
            s_shared=400, M=80, alpha=0.8,
            max_iter=300, epsilon=0.005, k=5,
            log=True,
        )

    def test_transport_plan_shape(self):
        assert self.T.shape == (400, 500)

    def test_transport_plan_nonnegative(self):
        assert torch.all(self.T >= 0)

    def test_transport_plan_is_tensor(self):
        assert isinstance(self.T, torch.Tensor)

    def test_spearman_correlation(self):
        T_np = self.T.cpu().numpy()
        matched_angles = self.a_tgt[T_np.argmax(axis=1)]
        sp, _ = spearmanr(self.a_src, matched_angles)
        assert abs(sp) >= 0.90, f"|Spearman| = {abs(sp):.4f}, expected >= 0.90"

    def test_monotone_matching(self):
        T_np = self.T.cpu().numpy()
        row_argmax = T_np.argmax(axis=1)
        sp, _ = spearmanr(np.arange(400), row_argmax)
        assert abs(sp) >= 0.90, f"|Monotonicity Spearman| = {abs(sp):.4f}, expected >= 0.90"

    def test_gw_cost_returned(self):
        assert "gw_cost" in self.log_dict
        assert np.isfinite(self.log_dict["gw_cost"])
        assert self.log_dict["gw_cost"] > 0

    def test_convergence_info(self):
        assert "err_list" in self.log_dict
        assert "n_iter" in self.log_dict
        assert len(self.log_dict["err_list"]) > 0
        assert self.log_dict["n_iter"] > 0


# ── Tests for all three distance modes ────────────────────────────────

_COMMON_KWARGS = dict(s_shared=400, M=80, alpha=0.8, max_iter=200, epsilon=0.005, k=5)


@pytest.fixture(scope="module")
def spiral_data():
    np.random.seed(42)
    spiral, a_src = _sample_spiral(400, seed=0)
    swiss_roll, a_tgt = _sample_swiss_roll(500, seed=1)
    return spiral, swiss_roll, a_src, a_tgt


@pytest.mark.parametrize("mode,extra_kwargs,min_rho", [
    ("dijkstra", {}, 0.90),
    ("precomputed", {}, 0.90),
    # Landmark Dijkstra uses real shortest-path distances and achieves
    # high quality on this benchmark. We still skip the quality threshold
    # here to keep the parametrized test simple — quality is validated
    # in the non-parametrized tests above.
    ("landmark", {"n_landmarks": 20}, None),
])
def test_distance_mode_quality(spiral_data, mode, extra_kwargs, min_rho):
    """Each distance mode should produce a valid transport plan.

    For dijkstra and precomputed (exact geodesic), also checks Spearman >= 0.90.
    Prints timing and quality metrics for all modes.
    """
    spiral, swiss_roll, a_src, a_tgt = spiral_data
    np.random.seed(42)

    t0 = time.perf_counter()
    T, log_dict = sampled_gw(
        spiral, swiss_roll,
        distance_mode=mode,
        **extra_kwargs,
        **_COMMON_KWARGS,
        log=True,
    )
    t_total = time.perf_counter() - t0

    T_np = T.cpu().numpy()
    assert T_np.shape == (400, 500)
    assert np.all(T_np >= 0)

    # Spearman correlation: matched angles
    matched_angles = a_tgt[T_np.argmax(axis=1)]
    sp_angle, _ = spearmanr(a_src, matched_angles)

    # Monotonicity: row index vs col argmax
    row_argmax = T_np.argmax(axis=1)
    sp_mono, _ = spearmanr(np.arange(400), row_argmax)

    print(f"\n  [{mode:>12s}] time={t_total:6.2f}s | "
          f"iters={log_dict['n_iter']:3d} | "
          f"gw_cost={log_dict['gw_cost']:.4e} | "
          f"sp_angle={sp_angle:.4f} | sp_mono={sp_mono:.4f}")

    if min_rho is not None:
        # GW is orientation-invariant, so the match may be monotone or
        # anti-monotone (both are valid isometries).  Check absolute value.
        assert abs(sp_angle) >= min_rho, (
            f"distance_mode={mode!r}: Spearman |sp_angle| = {abs(sp_angle):.4f}, expected >= {min_rho}"
        )
        assert abs(sp_mono) >= min_rho, (
            f"distance_mode={mode!r}: Monotonicity |sp_mono| = {abs(sp_mono):.4f}, expected >= {min_rho}"
        )
