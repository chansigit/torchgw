from __future__ import annotations

import torch


def _lr_dykstra(
    eps1: torch.Tensor,
    eps2: torch.Tensor,
    eps3: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dykstra's algorithm for projecting onto the low-rank coupling set.

    Alternates KL projections onto:
      C1: row marginals (Q @ 1 = a, R @ 1 = b, g >= alpha)
      C2: shared inner marginal (Q^T @ 1 = R^T @ 1 = g)

    Parameters
    ----------
    eps1 : (N, r) mirror descent iterate for Q
    eps2 : (K, r) mirror descent iterate for R
    eps3 : (r,) mirror descent iterate for g
    a : (N,) source marginal
    b : (K,) target marginal
    alpha : float, lower bound for g entries
    max_iter : int
    tol : float

    Returns
    -------
    Q : (N, r)
    R : (K, r)
    g : (r,)
    """
    r = eps3.shape[0]
    dtype = eps1.dtype
    device = eps1.device

    g_ = eps3.clone()
    q3_1 = torch.ones(r, dtype=dtype, device=device)
    q3_2 = torch.ones(r, dtype=dtype, device=device)
    v1_ = torch.ones(r, dtype=dtype, device=device)
    v2_ = torch.ones(r, dtype=dtype, device=device)
    q1 = torch.ones(r, dtype=dtype, device=device)
    q2 = torch.ones(r, dtype=dtype, device=device)

    for _ in range(max_iter):
        # --- Projection onto C1 (row marginals + g >= alpha) ---
        u1 = a / (eps1 @ v1_).clamp(min=1e-30)
        u2 = b / (eps2 @ v2_).clamp(min=1e-30)

        g = torch.maximum(torch.tensor(alpha, dtype=dtype, device=device), g_ * q3_1)
        q3_1 = (g_ * q3_1) / g.clamp(min=1e-30)
        g_ = g.clone()

        # --- Projection onto C2 (shared inner marginal) ---
        prod1 = (v1_ * q1) * (eps1.T @ u1)
        prod2 = (v2_ * q2) * (eps2.T @ u2)
        g = (g_ * q3_2 * prod1 * prod2) ** (1.0 / 3.0)
        g = g.clamp(min=1e-30)

        v1 = g / (eps1.T @ u1).clamp(min=1e-30)
        v2 = g / (eps2.T @ u2).clamp(min=1e-30)

        q1 = (v1_ * q1) / v1.clamp(min=1e-30)
        q2 = (v2_ * q2) / v2.clamp(min=1e-30)
        q3_2 = (g_ * q3_2) / g.clamp(min=1e-30)

        v1_ = v1.clone()
        v2_ = v2.clone()
        g_ = g.clone()

        # Convergence check
        err1 = torch.abs(u1 * (eps1 @ v1) - a).sum()
        err2 = torch.abs(u2 * (eps2 @ v2) - b).sum()
        if (err1 + err2).item() < tol:
            break

    Q = u1.unsqueeze(1) * eps1 * v1.unsqueeze(0)
    R = u2.unsqueeze(1) * eps2 * v2.unsqueeze(0)
    return Q, R, g


def sinkhorn_lowrank(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    rank: int,
    reg: float,
    max_iter: int = 5,
    tol: float = 1e-4,
    alpha: float = 1e-10,
    dykstra_max_iter: int = 50,
    dykstra_tol: float = 1e-7,
) -> torch.Tensor:
    """Low-rank Sinkhorn via mirror descent + Dykstra projection.

    Directly optimizes T = Q @ diag(1/g) @ R^T as low-rank factors,
    following Scetbon, Cuturi & Peyré (2021).

    The transport plan T has nonneg. rank <= r, with memory O((N+K)*r).

    Parameters
    ----------
    a : (N,) source marginal
    b : (K,) target marginal
    C : (N, K) cost matrix
    rank : int, target rank r
    reg : float, entropic regularization (>= 0)
    max_iter : int, outer mirror descent iterations
    tol : float, unused (kept for interface compat)
    alpha : float, lower bound for g (must be < 1/rank)
    dykstra_max_iter : int, max iterations for inner Dykstra
    dykstra_tol : float, convergence threshold for Dykstra

    Returns
    -------
    T : (N, K) transport plan (dense, reconstructed from factors)
    """
    N, K = C.shape
    r = min(rank, N, K)
    dtype = C.dtype
    device = C.device

    if 1.0 / r < alpha:
        alpha = 0.5 / r

    # ── Initialization (random, local generator to avoid polluting global state) ──
    gen = torch.Generator(device=device)
    gen.manual_seed(49)
    g = torch.abs(torch.randn(r, dtype=dtype, device=device, generator=gen)) + 1.0
    g = g / g.sum()

    Q = torch.abs(torch.randn(N, r, dtype=dtype, device=device, generator=gen)) + 1.0
    Q = Q * (a / Q.sum(dim=1).clamp(min=1e-30)).unsqueeze(1)

    R = torch.abs(torch.randn(K, r, dtype=dtype, device=device, generator=gen)) + 1.0
    R = R * (b / R.sum(dim=1).clamp(min=1e-30)).unsqueeze(1)

    # ── Outer mirror descent loop ──
    for _ in range(max_iter):
        diag_1g = (1.0 / g.clamp(min=1e-30)).unsqueeze(0)  # (1, r)

        # Cost-coupling products
        CR = C @ R                 # (N, r)
        CR_g = CR * diag_1g        # (N, r)

        CQ = C.T @ Q              # (K, r)
        CQ_g = CQ * diag_1g        # (K, r)

        omega = (Q * CR).sum(dim=0)  # (r,) = diag(Q^T @ C @ R)

        # Adaptive step size (rescale strategy)
        log_Q = torch.log(Q.clamp(min=1e-30))
        log_R = torch.log(R.clamp(min=1e-30))
        log_g = torch.log(g.clamp(min=1e-30))

        norm_1 = torch.max(torch.abs(CR_g + reg * log_Q)).item() ** 2
        norm_2 = torch.max(torch.abs(CQ_g + reg * log_R)).item() ** 2
        norm_3 = torch.max(torch.abs(-omega * diag_1g.squeeze(0))).item() ** 2
        gamma = 10.0 / max(norm_1, norm_2, norm_3, 1e-30)
        # Ensure gamma*reg >= 1 to prevent exponential blowup in mirror maps
        gamma = max(gamma, 1.0 / max(reg, 1e-30))

        # Mirror descent exponential maps
        eps1 = torch.exp(-gamma * CR_g - (gamma * reg - 1) * log_Q)
        eps2 = torch.exp(-gamma * CQ_g - (gamma * reg - 1) * log_R)
        eps3 = torch.exp(gamma * omega / g.clamp(min=1e-30) ** 2
                         - (gamma * reg - 1) * log_g)

        # Clamp for stability
        eps1 = eps1.clamp(min=1e-30, max=1e30)
        eps2 = eps2.clamp(min=1e-30, max=1e30)
        eps3 = eps3.clamp(min=1e-30, max=1e30)

        # Dykstra projection
        Q, R, g = _lr_dykstra(eps1, eps2, eps3, a, b, alpha,
                              dykstra_max_iter, dykstra_tol)

        # Numerical stabilization
        Q = Q + 1e-16
        R = R + 1e-16
        g = g + 1e-16

    # ── Reconstruct T = Q @ diag(1/g) @ R^T ──
    Qg = Q * (1.0 / g.clamp(min=1e-30)).unsqueeze(0)  # (N, r)
    T = Qg @ R.T                                         # (N, K)

    return T
