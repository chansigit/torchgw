"""Tests for Sinkhorn gradient modes (implicit, unrolled, approximate)."""
import torch
import pytest


def _sinkhorn_unrolled_ref(C, a, b, reg, n_iter=300):
    """Reference: unrolled Sinkhorn with full PyTorch autograd."""
    log_K = -C / reg
    log_a = torch.log(a)
    log_b = torch.log(b)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
    return torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))


class TestImplicitGrad:
    """Implicit differentiation should produce exact gradients."""

    def _get_grad(self, C0, a, b, reg, W, grad_mode):
        from torchgw._solver import _sinkhorn_differentiable
        C = C0.clone().requires_grad_(True)
        T = _sinkhorn_differentiable(C, a, b, reg, max_iter=300, tol=1e-8,
                                     grad_mode=grad_mode)
        (W * T).sum().backward()
        return C.grad.clone(), T.detach().clone()

    def test_implicit_matches_unrolled(self):
        """Implicit VJP should match unrolled autograd (both exact)."""
        torch.manual_seed(42)
        N, K = 8, 10
        C0 = torch.rand(N, K, dtype=torch.float64)
        a = torch.ones(N, dtype=torch.float64) / N
        b = torch.ones(K, dtype=torch.float64) / K
        reg, W = 0.1, torch.randn(N, K, dtype=torch.float64)

        grad_implicit, T_impl = self._get_grad(C0, a, b, reg, W, "implicit")

        C_unroll = C0.clone().requires_grad_(True)
        T_unroll = _sinkhorn_unrolled_ref(C_unroll, a, b, reg)
        (W * T_unroll).sum().backward()
        grad_unroll = C_unroll.grad.clone()

        assert torch.allclose(T_impl, T_unroll.detach(), atol=1e-6)
        rel_err = (grad_implicit - grad_unroll).norm() / grad_unroll.norm()
        assert rel_err < 0.02, f"Implicit vs unrolled rel_err={rel_err:.4f}"

    def test_implicit_finite_diff(self):
        """Implicit gradient should match finite differences."""
        torch.manual_seed(77)
        N, K = 6, 7
        C0 = torch.rand(N, K, dtype=torch.float64)
        a = torch.ones(N, dtype=torch.float64) / N
        b = torch.ones(K, dtype=torch.float64) / K
        reg, W = 0.1, torch.randn(N, K, dtype=torch.float64)

        grad_impl, _ = self._get_grad(C0, a, b, reg, W, "implicit")

        h = 1e-5
        rng = torch.Generator().manual_seed(123)
        for _ in range(10):
            i = torch.randint(0, N, (1,), generator=rng).item()
            j = torch.randint(0, K, (1,), generator=rng).item()
            C_p, C_m = C0.clone(), C0.clone()
            C_p[i, j] += h
            C_m[i, j] -= h
            T_p = _sinkhorn_unrolled_ref(C_p, a, b, reg, n_iter=300)
            T_m = _sinkhorn_unrolled_ref(C_m, a, b, reg, n_iter=300)
            g_fd = ((W * T_p).sum().item() - (W * T_m).sum().item()) / (2 * h)
            g_impl = grad_impl[i, j].item()
            denom = max(abs(g_fd), abs(g_impl), 1e-10)
            rel_err = abs(g_impl - g_fd) / denom
            assert rel_err < 0.05, (
                f"FD mismatch at ({i},{j}): impl={g_impl:.6e}, fd={g_fd:.6e}, "
                f"rel_err={rel_err:.4f}"
            )

    def test_implicit_descent_direction(self):
        """Gradient step using implicit grad should decrease loss."""
        torch.manual_seed(99)
        N, K = 10, 12
        C0 = torch.rand(N, K, dtype=torch.float64)
        a = torch.ones(N, dtype=torch.float64) / N
        b = torch.ones(K, dtype=torch.float64) / K
        reg, W = 0.1, torch.randn(N, K, dtype=torch.float64)

        grad_impl, T0 = self._get_grad(C0, a, b, reg, W, "implicit")
        loss0 = (W * T0).sum().item()

        C_new = C0 - 0.01 * grad_impl
        T_new = _sinkhorn_unrolled_ref(C_new, a, b, reg)
        loss1 = (W * T_new).sum().item()
        assert loss1 < loss0 - 1e-8, f"loss0={loss0:.6e}, loss1={loss1:.6e}"

    def test_implicit_nonuniform_marginals(self):
        """Implicit grad works with non-uniform a, b."""
        torch.manual_seed(55)
        N, K = 8, 10
        C0 = torch.rand(N, K, dtype=torch.float64)
        a = torch.softmax(torch.randn(N, dtype=torch.float64), dim=0)
        b = torch.softmax(torch.randn(K, dtype=torch.float64), dim=0)
        reg, W = 0.1, torch.randn(N, K, dtype=torch.float64)

        grad_impl, T_impl = self._get_grad(C0, a, b, reg, W, "implicit")

        C_unroll = C0.clone().requires_grad_(True)
        T_unroll = _sinkhorn_unrolled_ref(C_unroll, a, b, reg)
        (W * T_unroll).sum().backward()
        grad_unroll = C_unroll.grad.clone()

        rel_err = (grad_impl - grad_unroll).norm() / grad_unroll.norm()
        assert rel_err < 0.02, f"Non-uniform marginals: rel_err={rel_err:.4f}"


class TestUnrolledGrad:
    """grad_mode='unrolled' should match reference unrolled Sinkhorn."""

    def test_unrolled_matches_reference(self):
        torch.manual_seed(42)
        N, K = 8, 10
        C0 = torch.rand(N, K, dtype=torch.float64)
        a = torch.ones(N, dtype=torch.float64) / N
        b = torch.ones(K, dtype=torch.float64) / K
        reg, W = 0.1, torch.randn(N, K, dtype=torch.float64)

        from torchgw._solver import _sinkhorn_differentiable
        C1 = C0.clone().requires_grad_(True)
        T1 = _sinkhorn_differentiable(C1, a, b, reg, max_iter=300, tol=1e-8,
                                      grad_mode="unrolled")
        (W * T1).sum().backward()
        grad1 = C1.grad.clone()

        C2 = C0.clone().requires_grad_(True)
        T2 = _sinkhorn_unrolled_ref(C2, a, b, reg)
        (W * T2).sum().backward()
        grad2 = C2.grad.clone()

        assert torch.allclose(T1.detach(), T2.detach(), atol=1e-6)
        rel_err = (grad1 - grad2).norm() / grad2.norm()
        assert rel_err < 0.02, f"Unrolled mode vs reference: rel_err={rel_err:.4f}"


class TestApproximateGrad:
    """grad_mode='approximate' preserves existing frozen-potentials behavior."""

    def test_approximate_is_frozen_potentials(self):
        """Should reproduce the old grad_C = -grad_T * T / reg formula."""
        torch.manual_seed(42)
        N, K = 8, 10
        C0 = torch.rand(N, K, dtype=torch.float64)
        a = torch.ones(N, dtype=torch.float64) / N
        b = torch.ones(K, dtype=torch.float64) / K
        reg, W = 0.1, torch.randn(N, K, dtype=torch.float64)

        from torchgw._solver import _sinkhorn_differentiable
        C_req = C0.clone().requires_grad_(True)
        T = _sinkhorn_differentiable(C_req, a, b, reg, max_iter=300, tol=1e-8,
                                     grad_mode="approximate")
        (W * T).sum().backward()
        grad_approx = C_req.grad.clone()

        expected = -W * T.detach() / reg
        assert torch.allclose(grad_approx, expected, atol=1e-10), \
            "approximate mode should reproduce frozen-potentials formula"


class TestGradModeValidation:
    """Parameter validation for grad_mode."""

    def test_invalid_grad_mode(self):
        from torchgw._solver import _sinkhorn_differentiable
        N, K = 4, 5
        C = torch.rand(N, K, dtype=torch.float64)
        a = torch.ones(N, dtype=torch.float64) / N
        b = torch.ones(K, dtype=torch.float64) / K
        with pytest.raises(ValueError, match="grad_mode"):
            _sinkhorn_differentiable(C, a, b, 0.1, grad_mode="invalid")

    def test_default_grad_mode_is_implicit(self):
        """Default should be implicit (exact)."""
        torch.manual_seed(42)
        N, K = 6, 7
        C0 = torch.rand(N, K, dtype=torch.float64)
        a = torch.ones(N, dtype=torch.float64) / N
        b = torch.ones(K, dtype=torch.float64) / K
        reg, W = 0.1, torch.randn(N, K, dtype=torch.float64)

        from torchgw._solver import _sinkhorn_differentiable
        C1 = C0.clone().requires_grad_(True)
        T1 = _sinkhorn_differentiable(C1, a, b, reg, max_iter=300, tol=1e-8)
        (W * T1).sum().backward()
        grad_default = C1.grad.clone()

        C2 = C0.clone().requires_grad_(True)
        T2 = _sinkhorn_differentiable(C2, a, b, reg, max_iter=300, tol=1e-8,
                                      grad_mode="implicit")
        (W * T2).sum().backward()
        grad_implicit = C2.grad.clone()

        assert torch.allclose(grad_default, grad_implicit, atol=1e-12)
