import numpy as np
import torch
import pytest
from torchgw._solver import sampled_gw


def test_sampled_gw_returns_tensor(two_datasets):
    """Output should be a torch.Tensor in v0.2.0."""
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10)
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_sampled_gw_accepts_tensor_input(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        torch.from_numpy(X_src), torch.from_numpy(X_tgt),
        s_shared=50, M=30, max_iter=10,
    )
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)


def test_sampled_gw_mass_bounded(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10)
    assert T.sum().item() <= 1.0 + 1e-6


def test_distance_mode_dijkstra(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, distance_mode="dijkstra", s_shared=50, M=30, max_iter=10)
    assert T.shape == (150, 150)


def test_distance_mode_precomputed_with_matrices(two_datasets):
    from torchgw._graph import build_knn_graph
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    X_src, X_tgt = two_datasets
    g_src = build_knn_graph(X_src, k=10)
    g_tgt = build_knn_graph(X_tgt, k=10)
    C_X = sp_dijkstra(csgraph=g_src, directed=False).astype(np.float32)
    C_Y = sp_dijkstra(csgraph=g_tgt, directed=False).astype(np.float32)

    T = sampled_gw(
        dist_source=C_X, dist_target=C_Y,
        distance_mode="precomputed",
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_distance_mode_precomputed_without_matrices(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        distance_mode="precomputed",
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)


def test_distance_mode_landmark(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        distance_mode="landmark",
        n_landmarks=10,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_invalid_distance_mode(two_datasets):
    X_src, X_tgt = two_datasets
    with pytest.raises(ValueError, match="distance_mode"):
        sampled_gw(X_src, X_tgt, distance_mode="invalid", M=30, max_iter=10)


def test_precomputed_missing_params():
    with pytest.raises(ValueError):
        sampled_gw(distance_mode="precomputed", M=30, max_iter=10)


def test_sampled_gw_identity_alignment():
    """Self-alignment should produce a near-diagonal transport plan.

    GW on identical data can find symmetric (flipped) solutions, so we
    check that the argmax matching has high Spearman correlation with the
    identity, not exact diagonal hits.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 5)).astype(np.float32)
    T = sampled_gw(X, X.copy(), s_shared=50, M=50, max_iter=200, k=10,
                   alpha=0.5, epsilon=0.005)
    row_argmax = T.argmax(dim=1).cpu().numpy()
    sp, _ = spearmanr(np.arange(50), row_argmax)
    assert abs(sp) > 0.5, f"|Spearman| = {abs(sp):.4f}, expected > 0.5"


def test_sampled_gw_log_returns_tuple(two_datasets):
    X_src, X_tgt = two_datasets
    result = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10, log=True)
    assert isinstance(result, tuple)
    T, log_dict = result
    assert isinstance(T, torch.Tensor)
    assert "err_list" in log_dict
    assert "n_iter" in log_dict
    assert "gw_cost" in log_dict


def test_public_import():
    from torchgw import sampled_gw, build_knn_graph, joint_embedding
    assert callable(sampled_gw)
    assert callable(build_knn_graph)
    assert callable(joint_embedding)


def test_fused_gw(two_datasets):
    """Fused GW with fgw_alpha between 0 and 1."""
    X_src, X_tgt = two_datasets
    C_feat = torch.cdist(
        torch.from_numpy(X_src).float(),
        torch.from_numpy(X_tgt).float(),
    ) ** 2

    T = sampled_gw(
        X_src, X_tgt,
        fgw_alpha=0.5,
        C_linear=C_feat,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_pure_wasserstein(two_datasets):
    """fgw_alpha=1.0 should work without structural distances."""
    X_src, X_tgt = two_datasets
    C_feat = torch.cdist(
        torch.from_numpy(X_src).float(),
        torch.from_numpy(X_tgt).float(),
    ) ** 2

    T = sampled_gw(
        fgw_alpha=1.0,
        C_linear=C_feat,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_fgw_missing_c_linear(two_datasets):
    X_src, X_tgt = two_datasets
    with pytest.raises(ValueError, match="C_linear"):
        sampled_gw(X_src, X_tgt, fgw_alpha=0.5, M=30, max_iter=10)


def test_multiscale_basic(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, multiscale=True, s_shared=50, M=30, max_iter=10)
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_multiscale_with_precomputed(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        multiscale=True,
        distance_mode="precomputed",
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)


def test_multiscale_custom_n_coarse(two_datasets):
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, multiscale=True, n_coarse=30,
                   s_shared=50, M=30, max_iter=10)
    assert T.shape == (150, 150)


def test_lowrank_basic(two_datasets):
    from torchgw import sampled_lowrank_gw
    X_src, X_tgt = two_datasets
    T = sampled_lowrank_gw(X_src, X_tgt, rank=10, s_shared=50, M=30, max_iter=10)
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_lowrank_with_precomputed(two_datasets):
    from torchgw import sampled_lowrank_gw
    X_src, X_tgt = two_datasets
    T = sampled_lowrank_gw(
        X_src, X_tgt,
        rank=10,
        distance_mode="precomputed",
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)


def test_lowrank_and_multiscale(two_datasets):
    """Both features combined."""
    from torchgw import sampled_lowrank_gw
    X_src, X_tgt = two_datasets
    T = sampled_lowrank_gw(
        X_src, X_tgt,
        multiscale=True,
        rank=10,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_public_import_lowrank():
    from torchgw import sampled_lowrank_gw
    assert callable(sampled_lowrank_gw)


def test_differentiable_gradient_flows():
    """differentiable=True should allow gradients to flow through T."""
    X = torch.randn(20, 3)
    Y = torch.randn(25, 3)
    C_feat = torch.cdist(X, Y).requires_grad_(True)

    # Pure Wasserstein so Lambda = C_linear (differentiable end-to-end)
    T = sampled_gw(
        fgw_alpha=1.0, C_linear=C_feat,
        differentiable=True, M=10, max_iter=3, epsilon=0.1,
        device=torch.device("cpu"),
    )

    assert T.requires_grad, "T should require grad when differentiable=True"
    loss = (C_feat.detach() * T).sum()
    loss.backward()
    assert C_feat.grad is not None, "Gradient should flow to C_linear"
    assert C_feat.grad.shape == C_feat.shape


def test_differentiable_warning_pure_gw(two_datasets):
    """differentiable=True with fgw_alpha=0 should emit a warning."""
    import warnings
    X_src, X_tgt = two_datasets
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        T = sampled_gw(
            X_src, X_tgt,
            differentiable=True, fgw_alpha=0.0,
            M=10, max_iter=3, epsilon=0.1,
        )
        assert any("fgw_alpha=0" in str(warning.message) for warning in w), \
            "Should warn when differentiable=True with pure GW"


def test_differentiable_implicit_matches_unrolled():
    """sampled_gw with grad_mode='implicit' should match 'unrolled'."""
    torch.manual_seed(42)
    N, K = 8, 10
    C0 = torch.rand(N, K, dtype=torch.float64)
    W = torch.randn(N, K, dtype=torch.float64)
    kw = dict(fgw_alpha=1.0, differentiable=True, M=10, max_iter=5,
              epsilon=0.1, device=torch.device("cpu"))

    C1 = C0.clone().requires_grad_(True)
    T1 = sampled_gw(C_linear=C1, grad_mode="implicit", **kw)
    (W * T1).sum().backward()
    grad_implicit = C1.grad.clone()

    C2 = C0.clone().requires_grad_(True)
    T2 = sampled_gw(C_linear=C2, grad_mode="unrolled", **kw)
    (W * T2).sum().backward()
    grad_unrolled = C2.grad.clone()

    cos_sim = torch.nn.functional.cosine_similarity(
        grad_implicit.flatten(), grad_unrolled.flatten(), dim=0,
    )
    assert cos_sim > 0.95, f"implicit vs unrolled cosine_sim={cos_sim:.4f}"


def test_differentiable_approximate_is_inexact():
    """grad_mode='approximate' should have non-trivial error vs exact."""
    torch.manual_seed(42)
    N, K = 8, 10
    C0 = torch.rand(N, K, dtype=torch.float64)
    W = torch.randn(N, K, dtype=torch.float64)
    kw = dict(fgw_alpha=1.0, differentiable=True, M=10, max_iter=5,
              epsilon=0.1, device=torch.device("cpu"))

    C1 = C0.clone().requires_grad_(True)
    T1 = sampled_gw(C_linear=C1, grad_mode="implicit", **kw)
    (W * T1).sum().backward()
    grad_exact = C1.grad.clone()

    C2 = C0.clone().requires_grad_(True)
    T2 = sampled_gw(C_linear=C2, grad_mode="approximate", **kw)
    (W * T2).sum().backward()
    grad_approx = C2.grad.clone()

    rel_err = (grad_exact - grad_approx).norm() / grad_exact.norm()
    assert rel_err > 0.1, f"Expected non-trivial error, got {rel_err:.4f}"


def test_differentiable_gradient_fgw_blend():
    """Gradient should flow correctly through an FGW blend (0 < fgw_alpha < 1)."""
    torch.manual_seed(99)
    N, K, d = 40, 45, 3
    X = torch.randn(N, d, dtype=torch.float64)
    Y = torch.randn(K, d, dtype=torch.float64)
    C_feat = torch.cdist(X, Y).to(torch.float64).requires_grad_(True)

    T = sampled_gw(
        X.numpy(), Y.numpy(),
        fgw_alpha=0.5, C_linear=C_feat,
        differentiable=True, M=10, max_iter=5, epsilon=0.1,
        device=torch.device("cpu"),
    )
    loss = T.sum()
    loss.backward()
    assert C_feat.grad is not None, "Gradient should flow through FGW blend"
    assert not torch.all(C_feat.grad == 0), "Gradient should be non-trivial"


def test_differentiable_gradient_nonuniform_marginals():
    """Gradient should work with non-uniform marginals."""
    torch.manual_seed(55)
    N, K = 10, 12
    C = torch.rand(N, K, dtype=torch.float64).requires_grad_(True)
    p = torch.softmax(torch.randn(N, dtype=torch.float64), dim=0)
    q = torch.softmax(torch.randn(K, dtype=torch.float64), dim=0)

    T = sampled_gw(
        p=p, q=q,
        fgw_alpha=1.0, C_linear=C,
        differentiable=True, M=10, max_iter=5, epsilon=0.1,
        device=torch.device("cpu"),
    )
    loss = (C.detach() * T).sum()
    loss.backward()
    assert C.grad is not None
    assert C.grad.shape == C.shape
    assert not torch.all(C.grad == 0), "Gradient should be non-trivial"


def test_lambda_ema_basic(two_datasets):
    """lambda_ema_beta should produce a valid transport plan."""
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10,
                   lambda_ema_beta=0.5)
    assert isinstance(T, torch.Tensor)
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_lambda_ema_lowrank(two_datasets):
    """lambda_ema_beta should work with the low-rank solver too."""
    from torchgw import sampled_lowrank_gw
    X_src, X_tgt = two_datasets
    T = sampled_lowrank_gw(X_src, X_tgt, rank=10, s_shared=50, M=30,
                           max_iter=10, lambda_ema_beta=0.5)
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_lambda_ema_boundary_values(two_datasets):
    """beta=0.0 and beta=1.0 should be valid boundary cases."""
    X_src, X_tgt = two_datasets
    T0 = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=5,
                    lambda_ema_beta=0.0)
    T1 = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=5,
                    lambda_ema_beta=1.0)
    assert T0.shape == (150, 150)
    assert T1.shape == (150, 150)


def test_lambda_ema_invalid_values(two_datasets):
    """Out-of-range beta values should raise ValueError."""
    X_src, X_tgt = two_datasets
    with pytest.raises(ValueError, match="lambda_ema_beta"):
        sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=5,
                   lambda_ema_beta=1.5)
    with pytest.raises(ValueError, match="lambda_ema_beta"):
        sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=5,
                   lambda_ema_beta=-0.1)


def test_semi_relaxed(two_datasets):
    """Semi-relaxed mode should produce a valid plan with relaxed target marginal."""
    X_src, X_tgt = two_datasets
    T = sampled_gw(
        X_src, X_tgt,
        semi_relaxed=True, rho_a=1.0, rho_b=1.0,
        s_shared=50, M=30, max_iter=10,
    )
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)
    # Source marginal should still be approximately enforced
    p = torch.ones(150, dtype=T.dtype, device=T.device) / 150
    row_sums = T.sum(dim=1)
    assert torch.allclose(row_sums, p, atol=0.01), \
        f"Source marginal error: {(row_sums - p).abs().max():.4e}"


# ── mixed_precision tests ──────────────────────────────────────────────


def test_mixed_precision_basic(two_datasets):
    """mixed_precision=True should produce a valid plan with float64 output."""
    X_src, X_tgt = two_datasets
    T = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10,
                   mixed_precision=True)
    assert isinstance(T, torch.Tensor)
    assert T.dtype == torch.float64
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_mixed_precision_sinkhorn_consistent():
    """Float32 and float64 Sinkhorn on identical cost matrix should agree.

    We test at the Sinkhorn level (not full GW) to isolate precision from
    sampling noise — full GW runs draw different samples each time.
    """
    from torchgw._solver import _sinkhorn_torch
    N, K = 40, 50
    torch.manual_seed(42)
    C = torch.rand(N, K, dtype=torch.float64)
    a = torch.ones(N, dtype=torch.float64) / N
    b = torch.ones(K, dtype=torch.float64) / K

    T_fp64 = _sinkhorn_torch(a, b, C, reg=0.05, max_iter=100)
    T_fp32 = _sinkhorn_torch(a.float(), b.float(), C.float(), reg=0.05, max_iter=100)

    # Argmax agreement should be very high on identical input
    argmax_64 = T_fp64.argmax(dim=1)
    argmax_32 = T_fp32.float().argmax(dim=1)  # ensure same dtype for comparison
    agreement = (argmax_64 == argmax_32).float().mean().item()
    assert agreement > 0.9, f"Row argmax agreement only {agreement:.0%}"
    # Total mass should be nearly identical
    assert abs(T_fp64.sum().item() - T_fp32.double().sum().item()) < 0.01


def test_mixed_precision_with_log(two_datasets):
    """mixed_precision should work with log=True."""
    X_src, X_tgt = two_datasets
    result = sampled_gw(X_src, X_tgt, s_shared=50, M=30, max_iter=10,
                        mixed_precision=True, log=True)
    T, log_dict = result
    assert T.dtype == torch.float64
    assert np.isfinite(log_dict["gw_cost"])


def test_lowrank_mixed_precision(two_datasets):
    """mixed_precision should work with low-rank solver."""
    from torchgw import sampled_lowrank_gw
    X_src, X_tgt = two_datasets
    T = sampled_lowrank_gw(X_src, X_tgt, rank=10, s_shared=50, M=30,
                           max_iter=10, mixed_precision=True)
    assert T.dtype == torch.float64
    assert T.shape == (150, 150)
    assert torch.all(T >= 0)


def test_cost_plateau_early_stopping(two_datasets):
    """Solver should stop early when GW cost plateaus, not run to max_iter."""
    X_src, X_tgt = two_datasets
    _, log_high = sampled_gw(X_src, X_tgt, s_shared=50, M=30,
                             max_iter=500, min_iter_before_converge=20,
                             log=True)
    # With 500 max_iter but typical convergence, should stop well before 500
    assert log_high["n_iter"] < 500, (
        f"Expected early stopping but ran all {log_high['n_iter']} iterations"
    )
    assert log_high["n_iter"] >= 20  # must respect min_iter_before_converge


def test_lowrank_semi_relaxed_early_error():
    """semi_relaxed should raise immediately, before any computation."""
    from torchgw import sampled_lowrank_gw
    with pytest.raises(ValueError, match="semi_relaxed"):
        sampled_lowrank_gw(
            np.random.randn(10, 3).astype(np.float32),
            np.random.randn(10, 3).astype(np.float32),
            semi_relaxed=True, M=5, max_iter=5,
        )
