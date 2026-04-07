"""Unit tests for Triton fused Sinkhorn kernels.

All tests are skipped if CUDA is unavailable or Triton cannot be imported.
Each kernel is tested against its pure-PyTorch reference implementation.
"""
import torch
import pytest

_SKIP_REASON = "requires CUDA and Triton"


def _has_triton_cuda():
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


requires_triton = pytest.mark.skipif(not _has_triton_cuda(), reason=_SKIP_REASON)


# ── Helper: reference PyTorch implementations ───────────────────────

def _pytorch_row_update(log_K, log_v, log_a):
    return log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)


def _pytorch_col_update(log_K, log_u, log_b):
    return log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)


def _pytorch_marginal_err(log_K, log_u, log_v, a):
    marginal = torch.exp(log_u + torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1))
    return torch.abs(marginal - a).max().item()


def _pytorch_materialize_T(log_u, log_K, log_v):
    return torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))


# ── Row kernel tests ────────────────────────────────────────────────

@requires_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_row_kernel_matches_pytorch(dtype):
    """Triton row kernel should produce same log_u as PyTorch reference."""
    from torchgw._triton_sinkhorn import _sinkhorn_row_kernel
    import triton

    N, K = 64, 128
    torch.manual_seed(42)
    log_K = -torch.rand(N, K, device="cuda", dtype=dtype)
    log_v = torch.randn(K, device="cuda", dtype=dtype)
    log_a = torch.log(torch.ones(N, device="cuda", dtype=dtype) / N)

    # PyTorch reference
    ref = _pytorch_row_update(log_K, log_v, log_a)

    # Triton kernel
    log_u_triton = torch.empty(N, device="cuda", dtype=dtype)
    BLOCK_K = min(triton.next_power_of_2(K), 4096)
    _sinkhorn_row_kernel[(N,)](
        log_K, log_v, log_a, log_u_triton,
        N, K, log_K.stride(0), log_K.stride(1),
        BLOCK_K=BLOCK_K, USE_FP64=(dtype == torch.float64),
    )

    torch.testing.assert_close(log_u_triton, ref, atol=1e-5, rtol=1e-5)


# ── Column kernel tests ─────────────────────────────────────────────

@requires_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_col_kernel_matches_pytorch(dtype):
    """Triton col kernel should produce same log_v as PyTorch reference."""
    from torchgw._triton_sinkhorn import _sinkhorn_col_kernel
    import triton

    N, K = 64, 128
    torch.manual_seed(42)
    log_K = -torch.rand(N, K, device="cuda", dtype=dtype)
    log_u = torch.randn(N, device="cuda", dtype=dtype)
    log_b = torch.log(torch.ones(K, device="cuda", dtype=dtype) / K)

    ref = _pytorch_col_update(log_K, log_u, log_b)

    log_v_triton = torch.empty(K, device="cuda", dtype=dtype)
    BLOCK_N = min(triton.next_power_of_2(N), 4096)
    _sinkhorn_col_kernel[(K,)](
        log_K, log_u, log_b, log_v_triton,
        N, K, log_K.stride(0), log_K.stride(1),
        BLOCK_N=BLOCK_N, USE_FP64=(dtype == torch.float64),
    )

    torch.testing.assert_close(log_v_triton, ref, atol=1e-5, rtol=1e-5)


# ── Marginal error kernel tests ─────────────────────────────────────

@requires_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_marginal_err_kernel_matches_pytorch(dtype):
    """Triton marginal error kernel should match PyTorch computation."""
    from torchgw._triton_sinkhorn import _marginal_err_kernel
    import triton

    N, K = 64, 128
    torch.manual_seed(42)
    log_K = -torch.rand(N, K, device="cuda", dtype=dtype)
    log_u = torch.randn(N, device="cuda", dtype=dtype)
    log_v = torch.randn(K, device="cuda", dtype=dtype)
    a = torch.ones(N, device="cuda", dtype=dtype) / N

    ref_err = _pytorch_marginal_err(log_K, log_u, log_v, a)

    err_buf = torch.zeros(1, device="cuda", dtype=dtype)
    BLOCK_K = min(triton.next_power_of_2(K), 4096)
    _marginal_err_kernel[(N,)](
        log_K, log_u, log_v, a, err_buf,
        N, K, log_K.stride(0), log_K.stride(1),
        BLOCK_K=BLOCK_K, USE_FP64=(dtype == torch.float64),
    )

    triton_err = err_buf.item()
    # Use relative tolerance: values can be large, absolute diff is meaningless
    rel_diff = abs(triton_err - ref_err) / max(abs(ref_err), 1e-8)
    assert rel_diff < 1e-4, (
        f"Triton err={triton_err:.6e} vs PyTorch err={ref_err:.6e} (rel_diff={rel_diff:.2e})"
    )


# ── T materialization kernel tests ───────────────────────────────────

@requires_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_materialize_T_matches_pytorch(dtype):
    """Triton fused T materialization should match unfused PyTorch."""
    from torchgw._triton_sinkhorn import triton_materialize_T

    N, K = 64, 128
    torch.manual_seed(42)
    log_K = -torch.rand(N, K, device="cuda", dtype=dtype)
    log_u = torch.randn(N, device="cuda", dtype=dtype)
    log_v = torch.randn(K, device="cuda", dtype=dtype)

    ref = _pytorch_materialize_T(log_u, log_K, log_v)
    T_triton = triton_materialize_T(log_u, log_K, log_v)

    torch.testing.assert_close(T_triton, ref, atol=1e-5, rtol=1e-5)


# ── Full Sinkhorn loop: Triton vs PyTorch ───────────────────────────

@requires_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_triton_sinkhorn_loop_matches_pytorch(dtype):
    """triton_sinkhorn_loop should produce equivalent results to PyTorch fallback."""
    from torchgw._triton_sinkhorn import triton_sinkhorn_loop
    from torchgw._solver import _sinkhorn_loop_pytorch

    N, K = 40, 50
    torch.manual_seed(42)
    C = torch.rand(N, K, device="cuda", dtype=dtype)
    reg = 0.1
    log_K = -C / reg
    a = torch.ones(N, device="cuda", dtype=dtype) / N
    b = torch.ones(K, device="cuda", dtype=dtype) / K
    log_a = torch.log(a)
    log_b = torch.log(b)

    log_u_pt, log_v_pt = _sinkhorn_loop_pytorch(
        log_K, log_a, log_b, tau=1.0, max_iter=100, tol=1e-6,
        check_every=10, a=a,
    )
    log_u_tr, log_v_tr = triton_sinkhorn_loop(
        log_K, log_a, log_b, tau=1.0, max_iter=100, tol=1e-6,
        check_every=10, a=a,
    )

    # Both converge to the same Sinkhorn potentials
    T_pt = _pytorch_materialize_T(log_u_pt, log_K, log_v_pt)
    T_tr = _pytorch_materialize_T(log_u_tr, log_K, log_v_tr)
    torch.testing.assert_close(T_pt, T_tr, atol=1e-4, rtol=1e-4)


@requires_triton
def test_triton_sinkhorn_loop_unbalanced():
    """Triton loop should handle tau < 1 (semi-relaxed Sinkhorn)."""
    from torchgw._triton_sinkhorn import triton_sinkhorn_loop

    N, K = 30, 40
    torch.manual_seed(42)
    C = torch.rand(N, K, device="cuda", dtype=torch.float64)
    log_K = -C / 0.1
    a = torch.ones(N, device="cuda", dtype=torch.float64) / N
    b = torch.ones(K, device="cuda", dtype=torch.float64) / K
    log_a = torch.log(a)
    log_b = torch.log(b)

    log_u, log_v = triton_sinkhorn_loop(
        log_K, log_a, log_b, tau=0.5, max_iter=100, tol=1e-6,
        check_every=10, a=a,
    )
    assert torch.all(torch.isfinite(log_u))
    assert torch.all(torch.isfinite(log_v))


@requires_triton
def test_triton_sinkhorn_loop_warm_start():
    """Triton loop should accept warm-start potentials."""
    from torchgw._triton_sinkhorn import triton_sinkhorn_loop

    N, K = 30, 40
    torch.manual_seed(42)
    C = torch.rand(N, K, device="cuda", dtype=torch.float64)
    log_K = -C / 0.1
    a = torch.ones(N, device="cuda", dtype=torch.float64) / N
    b = torch.ones(K, device="cuda", dtype=torch.float64) / K
    log_a = torch.log(a)
    log_b = torch.log(b)

    # First solve
    log_u, log_v = triton_sinkhorn_loop(
        log_K, log_a, log_b, tau=1.0, max_iter=100, tol=1e-6,
        check_every=10, a=a,
    )
    # Warm-start second solve with slightly different cost
    C2 = C + 0.01 * torch.rand_like(C)
    log_K2 = -C2 / 0.1
    log_u2, log_v2 = triton_sinkhorn_loop(
        log_K2, log_a, log_b, tau=1.0, max_iter=100, tol=1e-6,
        check_every=10, a=a,
        log_u_init=log_u, log_v_init=log_v,
    )
    assert torch.all(torch.isfinite(log_u2))
    assert torch.all(torch.isfinite(log_v2))


# ── Non-square matrix ────────────────────────────────────────────────

@requires_triton
def test_triton_kernels_non_square():
    """Kernels should handle highly non-square matrices (N >> K)."""
    from torchgw._triton_sinkhorn import triton_sinkhorn_loop, triton_materialize_T

    N, K = 200, 30
    torch.manual_seed(42)
    C = torch.rand(N, K, device="cuda", dtype=torch.float64)
    log_K = -C / 0.1
    a = torch.ones(N, device="cuda", dtype=torch.float64) / N
    b = torch.ones(K, device="cuda", dtype=torch.float64) / K

    log_u, log_v = triton_sinkhorn_loop(
        log_K, torch.log(a), torch.log(b), tau=1.0,
        max_iter=100, tol=1e-6, check_every=10, a=a,
    )
    T = triton_materialize_T(log_u, log_K, log_v)
    assert T.shape == (N, K)
    assert torch.all(T >= 0)
    # Check marginals
    assert torch.allclose(T.sum(dim=1), a, atol=1e-4)
