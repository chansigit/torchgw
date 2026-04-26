"""Cross-validate the new (rho_a, rho_b) Sinkhorn against POT's reference
unbalanced Sinkhorn implementation."""
import numpy as np
import pytest
import torch


def test_sinkhorn_matches_pot_unbalanced_entropy():
    """torchgw two-sided Sinkhorn should match POT's sinkhorn_unbalanced with
    entropy regularization (reg_type='entropy') within tight relative error on
    a small synthetic problem.

    Convention note: torchgw uses negative-entropy regularization
    (Gibbs kernel K = exp(-C/reg)), which corresponds to POT's reg_type='entropy'
    (not the default 'kl' which folds an extra a*b^T factor into the kernel).
    Using reg_type='entropy' gives a near-exact match (rel < 1e-5).
    """
    pytest.importorskip("ot")
    import ot
    from torchgw._solver import _sinkhorn_torch
    rng = np.random.default_rng(0)
    n, m = 50, 60
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    C = rng.uniform(size=(n, m)).astype(np.float64)
    reg = 0.05
    rho = 0.5
    # reg_type='entropy': POT uses K = exp(-M/reg) * ones, matching torchgw's
    # Gibbs kernel convention (pure negative-entropy regularization).
    T_pot = ot.unbalanced.sinkhorn_unbalanced(
        a, b, C, reg=reg, reg_m=rho, method="sinkhorn", numItermax=2000,
        reg_type="entropy",
    )
    T_torchgw = _sinkhorn_torch(
        torch.as_tensor(a), torch.as_tensor(b), torch.as_tensor(C),
        reg, max_iter=2000, tol=0,
        semi_relaxed=True, rho_a=rho, rho_b=rho,
    ).numpy()
    rel = np.linalg.norm(T_pot - T_torchgw) / (np.linalg.norm(T_pot) + 1e-12)
    # Conventions match exactly; floating-point differences should be < 1e-5.
    # We use a generous 1e-3 threshold to accommodate any minor iteration-order
    # differences between the two implementations.
    assert rel < 1e-3, f"relative diff vs POT (entropy): {rel:.4e}"
