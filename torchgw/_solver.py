import warnings

import numpy as np
import torch

from torchgw._graph import build_knn_graph
from torchgw._sampling import sample_pairs_gpu
from torchgw._utils import get_device


# ── Sinkhorn core (shared by both no_grad and differentiable paths) ──────

def _sinkhorn_iterations(
    log_K: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    log_u: torch.Tensor,
    log_v: torch.Tensor,
    is_balanced: bool,
    tau: float,
    n_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure Sinkhorn iterations without convergence check (compilable)."""
    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v_raw = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        if is_balanced:
            log_v = log_v_raw
        else:
            log_v = tau * log_v_raw + (1 - tau) * log_v
    return log_u, log_v


# torch.compile for kernel fusion (lazy init to avoid import-time compilation)
_sinkhorn_iterations_compiled = None


def _get_compiled_sinkhorn():
    global _sinkhorn_iterations_compiled
    if _sinkhorn_iterations_compiled is None:
        _sinkhorn_iterations_compiled = torch.compile(
            _sinkhorn_iterations, mode="reduce-overhead", dynamic=False,
        )
    return _sinkhorn_iterations_compiled


def _sinkhorn_loop(
    log_K: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor,
    tau_a: float, tau_b: float, max_iter: int, tol: float, check_every: int,
    a: torch.Tensor, verbose: bool = False,
    log_u_init: torch.Tensor | None = None,
    log_v_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch order: Triton (CUDA only, balanced or single-tau semi-relaxed) →
    pure PyTorch fallback (handles both sides via tau_a, tau_b)."""
    fully_unbalanced = (tau_a != 1.0) and (tau_b != 1.0) and (tau_a != tau_b or tau_a < 1.0)
    if log_K.is_cuda and not fully_unbalanced:
        try:
            from torchgw._triton_sinkhorn import triton_sinkhorn_loop
            tau_legacy = tau_b  # legacy single-tau was on the v side
            return triton_sinkhorn_loop(log_K, log_a, log_b, tau_legacy, max_iter,
                                        tol, check_every, a, verbose,
                                        log_u_init=log_u_init, log_v_init=log_v_init)
        except (ImportError, RuntimeError):
            pass
    return _sinkhorn_loop_pytorch(log_K, log_a, log_b, tau_a, tau_b,
                                   max_iter, tol, check_every, a, verbose,
                                   log_u_init=log_u_init, log_v_init=log_v_init)


def _sinkhorn_loop_pytorch(
    log_K: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    tau_a: float,
    tau_b: float,
    max_iter: int,
    tol: float,
    check_every: int,
    a: torch.Tensor,
    verbose: bool = False,
    log_u_init: torch.Tensor | None = None,
    log_v_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch Sinkhorn fallback. tau_a, tau_b control KL damping per side
    (1.0 = strict balanced; <1 = unbalanced KL relaxation)."""
    log_u = log_u_init if log_u_init is not None else torch.zeros_like(log_a)
    log_v = log_v_init if log_v_init is not None else torch.zeros_like(log_b)
    is_balanced_a = (tau_a == 1.0)
    is_balanced_b = (tau_b == 1.0)

    for it in range(max_iter):
        log_u_raw = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_u = log_u_raw if is_balanced_a else tau_a * log_u_raw
        log_v_raw = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        log_v = log_v_raw if is_balanced_b else tau_b * log_v_raw

        if tol > 0 and (it + 1) % check_every == 0:
            log_marginal = log_u + torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            marginal_err = torch.abs(torch.exp(log_marginal) - a).max().item()
            if verbose:
                print(f"    sinkhorn {it+1:>4}/{max_iter} | marginal_err: {marginal_err:.4e}")
            if marginal_err < tol:
                if verbose:
                    print(f"    sinkhorn converged at {it+1} (err={marginal_err:.4e})")
                break
    return log_u, log_v


def _sinkhorn_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    reg: float,
    max_iter: int = 100,
    tol: float = 5e-4,
    check_every: int = 10,
    semi_relaxed: bool = False,
    rho_a: float = 1.0,
    rho_b: float = 1.0,
    _inplace_C: bool = False,
    verbose: bool = False,
    log_u_init: torch.Tensor | None = None,
    log_v_init: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-domain Sinkhorn supporting balanced, single-side semi-relaxed,
    and fully-unbalanced via (rho_a, rho_b)."""
    log_K = C.neg_().div_(reg) if _inplace_C else -C / reg
    log_a = torch.log(a.clamp(min=1e-30))
    log_b = torch.log(b.clamp(min=1e-30))
    if semi_relaxed:
        tau_a = rho_a / (rho_a + reg)
        tau_b = rho_b / (rho_b + reg)
    else:
        tau_a = tau_b = 1.0

    log_u, log_v = _sinkhorn_loop(log_K, log_a, log_b, tau_a, tau_b,
                                   max_iter, tol, check_every, a,
                                   verbose=verbose,
                                   log_u_init=log_u_init, log_v_init=log_v_init)

    # Fused T materialization (Triton on CUDA, PyTorch fallback)
    if log_K.is_cuda:
        try:
            from torchgw._triton_sinkhorn import triton_materialize_T
            T = triton_materialize_T(log_u, log_K, log_v)
        except (ImportError, RuntimeError):
            T = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))
    else:
        T = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))

    # Stash potentials for warm-starting the next call
    T._log_u = log_u.detach()  # type: ignore[attr-defined]
    T._log_v = log_v.detach()  # type: ignore[attr-defined]
    return T


def _adjoint_sinkhorn_vjp(
    T: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float,
    grad_T: torch.Tensor,
) -> torch.Tensor:
    """Compute dL/dC via implicit differentiation at the Sinkhorn fixed point.

    The adjoint system (from IFT on Sinkhorn fixed-point conditions) is:

        J^T · [λ_u, λ_v] = [r_u, r_v]

    where J = [[I, P], [R, I]], P_{ij} = T_{ij}/a_i, R_{ji} = T_{ij}/b_j,
    r_u = (G ⊙ T)·1, r_v = (G ⊙ T)^T·1, G = grad_T.

    Solved via Schur complement on J^T (well-conditioned, eigenvalues in
    [0, 2]).  The system has a rank-1 null space from the Sinkhorn potential
    constant ambiguity, removed by a rank-1 correction (11^T/K) that
    preserves the gradient-relevant components.

    Final VJP: dL/dC_{kl} = (T_{kl}/ε) · (-G_{kl} + λ_u_k/a_k + λ_v_l/b_l)
    """
    N, K = T.shape
    G_T = grad_T * T  # G ⊙ T, shape (N, K)
    r_u = G_T.sum(dim=1)  # (N,)
    r_v = G_T.sum(dim=0)  # (K,)

    # Jacobian blocks (row-stochastic matrices)
    P = T / a.unsqueeze(1)   # (N, K), P_{ij} = T_{ij}/a_i
    R_T = T / b.unsqueeze(0) # (N, K), R^T_{ij} = T_{ij}/b_j

    # Schur complement on J^T: eliminate λ_u = r_u - R^T λ_v
    # → (I_K - P^T R^T) λ_v = r_v - P^T r_u
    S = torch.eye(K, dtype=T.dtype, device=T.device) - P.T @ R_T  # (K, K)

    # S has a rank-1 null space (eigvec ∝ 1_K) from the potential constant
    # ambiguity.  Adding 11^T/K replaces the zero eigenvalue with 1, making
    # S nonsingular.  Since the RHS is orthogonal to 1_K (sum(r_u) = sum(r_v)
    # for any valid upstream gradient), this doesn't affect the solution.
    S += torch.ones(K, K, dtype=T.dtype, device=T.device) / K

    rhs_v = r_v - P.T @ r_u                # (K,)
    lambda_v = torch.linalg.solve(S, rhs_v) # (K,)
    lambda_u = r_u - R_T @ lambda_v          # (N,)

    grad_C = (T / reg) * (-grad_T
                           + (lambda_u / a).unsqueeze(1)
                           + (lambda_v / b).unsqueeze(0))
    return grad_C


class _SinkhornImplicit(torch.autograd.Function):
    """Differentiable Sinkhorn with exact gradient via implicit differentiation."""

    @staticmethod
    def forward(ctx, C, a, b, reg, max_iter, tol, check_every):
        log_K = -C / reg
        log_a = torch.log(a.clamp(min=1e-30))
        log_b = torch.log(b.clamp(min=1e-30))

        log_u, log_v = _sinkhorn_loop(log_K, log_a, log_b, 1.0, 1.0,
                                       max_iter, tol, check_every, a)
        T = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))

        ctx.save_for_backward(T, a, b)
        ctx.reg = reg
        return T

    @staticmethod
    def backward(ctx, grad_T):
        T, a, b = ctx.saved_tensors
        grad_C = _adjoint_sinkhorn_vjp(T, a, b, ctx.reg, grad_T)
        return grad_C, None, None, None, None, None, None


class _SinkhornApproximate(torch.autograd.Function):
    """Differentiable Sinkhorn with frozen-potentials approximation.

    Backward: dT/dC ≈ -T/ε (treats potentials as constants).
    Fast but inexact — use _SinkhornImplicit for exact gradients.
    """

    @staticmethod
    def forward(ctx, C, a, b, reg, max_iter, tol, check_every, semi_relaxed, rho_a, rho_b):
        if semi_relaxed:
            tau_a = rho_a / (rho_a + reg)
            tau_b = rho_b / (rho_b + reg)
        else:
            tau_a = tau_b = 1.0
        log_K = -C / reg
        log_a = torch.log(a.clamp(min=1e-30))
        log_b = torch.log(b.clamp(min=1e-30))
        log_u, log_v = _sinkhorn_loop(log_K, log_a, log_b, tau_a, tau_b,
                                       max_iter, tol, check_every, a, False)
        T = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))
        ctx.save_for_backward(T, a, b)
        ctx.reg = reg
        return T

    @staticmethod
    def backward(ctx, grad_T):
        (T, a, b) = ctx.saved_tensors
        grad_C = -grad_T * T / ctx.reg
        return grad_C, None, None, None, None, None, None, None, None, None


def _sinkhorn_unrolled(
    C, a, b, reg, max_iter=100, tol=5e-4, check_every=10,
    semi_relaxed=False, rho_a: float = 1.0, rho_b: float = 1.0,
    grad_mode="autograd", verbose=False,
):
    if semi_relaxed:
        tau_a = rho_a / (rho_a + reg)
        tau_b = rho_b / (rho_b + reg)
    else:
        tau_a = tau_b = 1.0
    log_K = -C / reg
    log_a = torch.log(a.clamp(min=1e-30))
    log_b = torch.log(b.clamp(min=1e-30))
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    is_a_balanced = (tau_a == 1.0)
    is_b_balanced = (tau_b == 1.0)
    for it in range(max_iter):
        log_u_raw = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_u = log_u_raw if is_a_balanced else tau_a * log_u_raw
        log_v_raw = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        log_v = log_v_raw if is_b_balanced else tau_b * log_v_raw
        if tol > 0 and (it + 1) % check_every == 0:
            log_marginal = log_u + torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            if torch.abs(torch.exp(log_marginal) - a).max().item() < tol:
                break
    return torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))


_VALID_GRAD_MODES = {"implicit", "unrolled", "approximate"}


def _sinkhorn_differentiable(
    C, a, b, reg, max_iter=100, tol=5e-4, check_every=10,
    semi_relaxed=False, rho_a: float = 1.0, rho_b: float = 1.0,
    grad_mode="autograd", verbose=False,
):
    if semi_relaxed:
        return _sinkhorn_unrolled(C, a, b, reg, max_iter, tol, check_every,
                                  semi_relaxed, rho_a, rho_b, grad_mode, verbose)
    if grad_mode == "implicit":
        return _SinkhornImplicit.apply(C, a, b, reg, max_iter, tol, check_every)
    if grad_mode == "approximate":
        return _SinkhornApproximate.apply(
            C, a, b, reg, max_iter, tol, check_every, semi_relaxed, rho_a, rho_b,
        )
    return _sinkhorn_unrolled(C, a, b, reg, max_iter, tol, check_every,
                              semi_relaxed, rho_a, rho_b, grad_mode, verbose)


# ── Input coercion ──────────────────────────────────────────────────────

def _to_tensor(x):
    """Convert numpy array or tensor to torch.Tensor. Pass through None."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


# ── Shared preprocessing ────────────────────────────────────────────────

def _prepare_inputs(
    X_source, X_target, p, q, dist_source, dist_target, C_linear,
    distance_mode, fgw_alpha, k, n_landmarks, device,
):
    """Coerce inputs, infer N/K, validate, build distance provider.

    Returns (X_source, X_target, dist_source, dist_target, C_linear_t,
             N, K, provider, device).
    """
    from torchgw._distances import DijkstraProvider, PrecomputedProvider, LandmarkProvider

    X_source = _to_tensor(X_source)
    X_target = _to_tensor(X_target)
    p = _to_tensor(p)
    q = _to_tensor(q)
    dist_source = _to_tensor(dist_source)
    dist_target = _to_tensor(dist_target)
    C_linear_t = _to_tensor(C_linear)

    # Infer N, K
    if dist_source is not None and dist_target is not None:
        N, K = dist_source.shape[0], dist_target.shape[0]
    elif X_source is not None and X_target is not None:
        N, K = X_source.shape[0], X_target.shape[0]
    elif C_linear_t is not None:
        N, K = C_linear_t.shape
    else:
        raise ValueError(
            "Cannot infer dataset sizes. Provide (X_source, X_target), "
            "(dist_source, dist_target), or C_linear."
        )

    # Validate
    _VALID_MODES = {"precomputed", "dijkstra", "landmark"}
    if distance_mode not in _VALID_MODES:
        raise ValueError(
            f"distance_mode must be one of {_VALID_MODES}, got {distance_mode!r}"
        )
    if fgw_alpha > 0 and C_linear_t is None:
        raise ValueError("fgw_alpha > 0 requires C_linear to be provided")

    if device is None:
        device = get_device()

    # Build distance provider
    if fgw_alpha >= 1.0:
        provider = None
    elif distance_mode == "precomputed":
        if dist_source is not None and dist_target is not None:
            provider = PrecomputedProvider(dist_source=dist_source, dist_target=dist_target)
        elif X_source is not None and X_target is not None:
            graphs = (
                build_knn_graph(X_source.cpu().numpy(), k=k),
                build_knn_graph(X_target.cpu().numpy(), k=k),
            )
            provider = PrecomputedProvider(graph_source=graphs[0], graph_target=graphs[1])
        else:
            raise ValueError(
                "distance_mode='precomputed' requires (dist_source, dist_target) "
                "or (X_source, X_target)"
            )
    elif distance_mode == "dijkstra":
        if X_source is None or X_target is None:
            raise ValueError("distance_mode='dijkstra' requires X_source and X_target")
        graph_source = build_knn_graph(X_source.cpu().numpy(), k=k)
        graph_target = build_knn_graph(X_target.cpu().numpy(), k=k)
        provider = DijkstraProvider(graph_source, graph_target)
    elif distance_mode == "landmark":
        if X_source is None or X_target is None:
            raise ValueError("distance_mode='landmark' requires X_source and X_target")
        graph_source = build_knn_graph(X_source.cpu().numpy(), k=k)
        graph_target = build_knn_graph(X_target.cpu().numpy(), k=k)
        provider = LandmarkProvider(graph_source, graph_target, n_landmarks=n_landmarks)

    return X_source, X_target, p, q, dist_source, dist_target, C_linear_t, N, K, provider, device


def _gw_loop(
    *,
    N: int,
    K: int,
    provider,
    p_real: torch.Tensor,
    q_real: torch.Tensor,
    T_init: torch.Tensor,
    sinkhorn_fn,
    use_augmented: bool,
    s_shared: int | None,
    fgw_alpha: float,
    C_lin_device: torch.Tensor | None,
    M: int,
    alpha: float,
    max_iter: int,
    tol: float,
    epsilon: float,
    min_iter_before_converge: int,
    device: torch.device,
    verbose: bool,
    verbose_every: int,
    semi_relaxed: bool,
    rho_a: float,
    rho_b: float,
    differentiable: bool = False,
    lambda_ema_beta: float | None = None,
    mixed_precision: bool = False,
) -> tuple[torch.Tensor, list, int, float]:
    """Shared main loop for sampled_gw and sampled_lowrank_gw.

    Parameters
    ----------
    sinkhorn_fn : callable
        For standard: takes (p_aug, q_aug, Lambda_aug, reg, **kw) -> T_aug
        For low-rank: takes (p_real, q_real, Lambda, reg, **kw) -> T_new
    use_augmented : bool
        If True, build augmented cost/marginals and call sinkhorn_fn on them.
        If False, call sinkhorn_fn directly on (p_real, q_real, Lambda).

    Returns
    -------
    T_out, err_list, n_iter, gw_cost_val
    """
    if lambda_ema_beta is not None and not (0.0 <= lambda_ema_beta <= 1.0):
        raise ValueError(f"lambda_ema_beta must be in [0, 1], got {lambda_ema_beta}")
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")

    # Sinkhorn internal dtype: float32 when mixed_precision, else float64
    sink_dtype = torch.float32 if mixed_precision else torch.float64

    T_real = T_init.to(sink_dtype)

    # Augmented marginals (only needed for standard Sinkhorn)
    if use_augmented:
        m_frac = s_shared / max(N, K) if s_shared is not None else min(N, K) / max(N, K)
        slack_p = max(q_real.sum().item() - m_frac, 1e-10)
        slack_q = max(p_real.sum().item() - m_frac, 1e-10)
        p_aug = torch.cat([p_real, torch.tensor([slack_p], device=device, dtype=torch.float64)])
        q_aug = torch.cat([q_real, torch.tensor([slack_q], device=device, dtype=torch.float64)])

    # Regularization decay (at most 10x reduction to avoid instability)
    initial_reg = epsilon if epsilon > 0 else 1e-4
    final_reg = max(initial_reg / 10.0, min(5e-4, initial_reg))
    decay = (final_reg / initial_reg) ** (1 / max(1, max_iter))

    err_list = []
    gw_cost_val = 0.0
    n_iter = 0
    Lambda_ema = None  # EMA state for cost matrix smoothing
    _warm_log_u: torch.Tensor | None = None  # Sinkhorn warm-start potentials
    _warm_log_v: torch.Tensor | None = None

    # Cost plateau detection via EMA + patience.
    # Critical because err = ||T - T_prev|| reflects sampling noise (not
    # optimization progress) and may never converge to tol.
    _cost_ema: float | None = None
    _best_cost_ema = float('inf')
    _no_improve = 0
    _patience = max(min_iter_before_converge // 2, 20)

    # Pre-allocate augmented cost matrix and cast marginals (reused every iteration)
    if use_augmented:
        Lambda_aug = torch.zeros(N + 1, K + 1, device=device, dtype=sink_dtype)
        p_aug_sink = p_aug.to(sink_dtype)
        q_aug_sink = q_aug.to(sink_dtype)
    p_sink = p_real.to(sink_dtype)
    q_sink = q_real.to(sink_dtype)

    for i in range(max_iter):
        current_reg = initial_reg * (decay ** i)

        # Sample anchor pairs (on GPU, only transfers 2*M ints back)
        j_left, l_target = sample_pairs_gpu(T_real.detach(), M)

        # Compute distances via provider
        if provider is not None:
            D_left, D_tgt = provider.get_distances(j_left, l_target, device)

            for D in [D_left, D_tgt]:
                inf_mask = torch.isinf(D)
                if torch.any(inf_mask):
                    finite_vals = D[~inf_mask]
                    fill = finite_vals.max() * 1.5 if finite_vals.numel() > 0 else 1.0
                    D[inf_mask] = fill
                mx = D.max()
                if mx > 0:
                    D /= mx

            # Build Lambda_gw in sink_dtype (float32 when mixed_precision)
            # to avoid a full N*K float64 allocation.
            D_left_s = D_left if D_left.dtype == sink_dtype else D_left.to(sink_dtype)
            D_tgt_s = D_tgt if D_tgt.dtype == sink_dtype else D_tgt.to(sink_dtype)
            term_A = torch.mean(D_left_s ** 2, dim=1, keepdim=True)
            term_C = torch.mean(D_tgt_s ** 2, dim=1, keepdim=True).T
            Lambda_gw = torch.mm(D_left_s, D_tgt_s.T)
            Lambda_gw.mul_(-2.0 / M).add_(term_A).add_(term_C)
            del D_left_s, D_tgt_s

            # Lambda EMA: smooth cost matrix across iterations
            # beta=0.0 is treated as disabled (same as None)
            if lambda_ema_beta is not None and lambda_ema_beta > 0:
                if Lambda_ema is None:
                    Lambda_ema = Lambda_gw
                else:
                    Lambda_ema = (1 - lambda_ema_beta) * Lambda_ema + lambda_ema_beta * Lambda_gw
                Lambda_gw = Lambda_ema.clone()
        else:
            Lambda_gw = None

        # FGW blending
        if fgw_alpha >= 1.0:
            Lambda = C_lin_device
        elif fgw_alpha > 0:
            Lambda = (1 - fgw_alpha) * Lambda_gw + fgw_alpha * C_lin_device
        else:
            Lambda = Lambda_gw

        # Sinkhorn step
        if use_augmented:
            Lambda_aug[:N, :K] = Lambda if Lambda.dtype == sink_dtype else Lambda.to(sink_dtype)
            penalty = 100.0 * Lambda.max().clamp(min=1.0)  # stays on GPU, no sync
            Lambda_aug[:-1, -1] = penalty
            Lambda_aug[-1, :-1] = penalty
            Lambda_aug[-1, -1] = 0.0

            verbose_sink = verbose and (n_iter + 1) % verbose_every == 0
            T_aug = sinkhorn_fn(p_aug_sink, q_aug_sink,
                                Lambda_aug, current_reg,
                                semi_relaxed=semi_relaxed, rho_a=rho_a, rho_b=rho_b,
                                verbose=verbose_sink,
                                log_u_init=_warm_log_u, log_v_init=_warm_log_v,
                                _inplace_C=True)
            T_new = T_aug[:-1, :-1]
            # Retrieve potentials for warm-starting next iteration
            _warm_log_u = getattr(T_aug, '_log_u', None)
            _warm_log_v = getattr(T_aug, '_log_v', None)
        else:
            verbose_sink = verbose and (n_iter + 1) % verbose_every == 0
            Lambda_sink = Lambda if Lambda.dtype == sink_dtype else Lambda.to(sink_dtype)
            T_new = sinkhorn_fn(p_sink, q_sink,
                                Lambda_sink,
                                current_reg, semi_relaxed=semi_relaxed, rho_a=rho_a, rho_b=rho_b,
                                verbose=verbose_sink,
                                log_u_init=_warm_log_u, log_v_init=_warm_log_v)
            _warm_log_u = getattr(T_new, '_log_u', None)
            _warm_log_v = getattr(T_new, '_log_v', None)

        # In-place momentum update: T_real = (1-alpha)*T_real + alpha*T_new
        # Avoids allocating a separate T_prev copy (saves one N*K buffer).
        # Compute convergence metric BEFORE the in-place update.
        if differentiable:
            # Differentiable mode: keep T_new in graph, no in-place
            T_prev = T_real.detach().clone()
            T_real = (1 - alpha) * T_prev + alpha * T_new
            err_tensor = torch.linalg.norm(T_real - T_prev)
            del T_prev
        else:
            # In-place momentum: T_real ← (1-α)T_real + αT_new
            # After update: T_real_new - T_real_old = α(T_new - T_real_old)
            #   = α/(1-α) * (T_real_new - T_new)   [since T_real_new - T_new = (1-α)(T_real_old - T_new)]
            # Compute in-place to avoid N*K temporaries:
            T_real.mul_(1 - alpha).add_(T_new, alpha=alpha)
            # Reuse T_new buffer for err: T_new ← T_real - T_new (in-place)
            T_new.neg_().add_(T_real)          # T_new is now (T_real_new - T_new_orig)
            err_tensor = (alpha / (1 - alpha)) * T_new.norm()

        n_iter = i + 1
        _check_interval = 5  # sync with CPU every N iterations

        # Only sync to CPU at check intervals (reduces CUDA sync overhead)
        if n_iter % _check_interval == 0 or i == max_iter - 1 or i >= min_iter_before_converge:
            # Frobenius inner product <Lambda, T>, computed via batched row
            # dots to avoid N*K temporary and int32 overflow (N*K > 2^31).
            Lambda_s = Lambda if Lambda.dtype == sink_dtype else Lambda.to(sink_dtype)
            gw_cost_val = torch.bmm(
                Lambda_s.unsqueeze(1), T_real.unsqueeze(2)
            ).sum().item()
            err = err_tensor.item()
            err_list.append(err)

            if verbose and (n_iter % verbose_every == 0 or i == max_iter - 1):
                print(f"  iter {n_iter:>4}/{max_iter} | err: {err:.4e} | "
                      f"gw_cost: {gw_cost_val:.4e} | reg: {current_reg:.4e}")

            # Convergence: plan change OR cost EMA plateau
            _cost_ema = gw_cost_val if _cost_ema is None else 0.8 * _cost_ema + 0.2 * gw_cost_val
            if i >= min_iter_before_converge:
                if err < tol:
                    if verbose:
                        print(f"  converged at iteration {n_iter} (err={err:.4e})")
                    break
                if _cost_ema < _best_cost_ema * 0.995:
                    _best_cost_ema = _cost_ema
                    _no_improve = 0
                else:
                    _no_improve += 1
                if _no_improve >= _patience:
                    if verbose:
                        print(f"  cost plateau at iteration {n_iter} "
                              f"(no improve for {_patience} iters, gw_cost={gw_cost_val:.4e})")
                    break

        del T_new
        if use_augmented:
            del T_aug
        if provider is not None:
            del D_left, D_tgt, Lambda_gw

    # Cast back to float64 for output precision
    if T_real.dtype != torch.float64:
        try:
            T_out = T_real.to(torch.float64)
        except torch.cuda.OutOfMemoryError:
            T_out = T_real  # keep float32 if float64 copy would OOM
    else:
        T_out = T_real
    if differentiable:
        return T_out, err_list, n_iter, gw_cost_val
    return T_out.detach(), err_list, n_iter, gw_cost_val


# ── Multiscale helper ────────────────────────────────────────────────────

def _maybe_multiscale(
    multiscale, n_coarse, X_source, X_target, N, K,
    dist_source, dist_target, C_linear_t, fgw_alpha,
    distance_mode, n_landmarks, device, p_real, q_real,
    solver_fn, solver_kwargs,
):
    """Run coarse solve and return upsampled T_init, or None."""
    if not multiscale or X_source is None or X_target is None:
        return None

    from torchgw._multiscale import fps_downsample, upsample_plan

    _n_coarse = n_coarse if n_coarse is not None else min(500, N // 4, K // 4)
    _n_coarse = max(_n_coarse, 10)

    if _n_coarse >= N or _n_coarse >= K:
        return None

    idx_src, assign_src = fps_downsample(X_source, _n_coarse)
    idx_tgt, assign_tgt = fps_downsample(X_target, _n_coarse)

    X_src_coarse = X_source[idx_src]
    X_tgt_coarse = X_target[idx_tgt]

    C_lin_coarse = None
    if C_linear_t is not None and fgw_alpha > 0:
        C_lin_coarse = C_linear_t[idx_src][:, idx_tgt]

    dist_src_coarse = None
    dist_tgt_coarse = None
    if dist_source is not None and dist_target is not None:
        dist_src_coarse = dist_source[idx_src][:, idx_src]
        dist_tgt_coarse = dist_target[idx_tgt][:, idx_tgt]

    coarse_kwargs = {**solver_kwargs}
    coarse_kwargs['multiscale'] = False  # no recursion
    coarse_kwargs['M'] = min(solver_kwargs.get('M', 50), max(_n_coarse // 2, 10))
    coarse_kwargs['k'] = min(solver_kwargs.get('k', 30), _n_coarse - 1)
    if solver_kwargs.get('s_shared') is not None:
        coarse_kwargs['s_shared'] = min(solver_kwargs['s_shared'], _n_coarse)

    T_coarse = solver_fn(
        X_src_coarse, X_tgt_coarse,
        distance_mode=distance_mode,
        dist_source=dist_src_coarse, dist_target=dist_tgt_coarse,
        n_landmarks=n_landmarks,
        fgw_alpha=fgw_alpha, C_linear=C_lin_coarse,
        device=device,
        **coarse_kwargs,
    )

    T_init = upsample_plan(T_coarse, assign_src, assign_tgt, p_real, q_real)
    return T_init


# ── Public API: standard solver ─────────────────────────────────────────

def sampled_gw(
    X_source: np.ndarray | torch.Tensor | None = None,
    X_target: np.ndarray | torch.Tensor | None = None,
    p: np.ndarray | torch.Tensor | None = None,
    q: np.ndarray | torch.Tensor | None = None,
    *,
    distance_mode: str = "dijkstra",
    dist_source: np.ndarray | torch.Tensor | None = None,
    dist_target: np.ndarray | torch.Tensor | None = None,
    n_landmarks: int = 50,
    fgw_alpha: float = 0.0,
    C_linear: np.ndarray | torch.Tensor | None = None,
    s_shared: int | None = None,
    M: int = 50,
    alpha: float = 0.9,
    max_iter: int = 500,
    tol: float = 1e-5,
    epsilon: float = 0.001,
    k: int = 30,
    min_iter_before_converge: int = 50,
    device: torch.device | None = None,
    verbose: bool = False,
    verbose_every: int = 20,
    log: bool = False,
    differentiable: bool = False,
    grad_mode: str = "implicit",
    semi_relaxed: bool = False,
    rho_a: float = 1.0,
    rho_b: float = 1.0,
    multiscale: bool = False,
    n_coarse: int | None = None,
    lambda_ema_beta: float | None = None,
    mixed_precision: bool = False,
    T_init: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Sampled Gromov-Wasserstein alignment between two datasets.

    Uses standard log-domain Sinkhorn with slack augmentation for
    partial transport.

    Parameters
    ----------
    X_source, X_target : ndarray or Tensor, optional
        Feature matrices. Required unless dist matrices or C_linear provided.
    p, q : ndarray or Tensor, optional
        Marginal distributions (uniform if None).
    distance_mode : str
        ``"dijkstra"`` (default), ``"precomputed"``, or ``"landmark"``.
    dist_source, dist_target : ndarray or Tensor, optional
        Precomputed (ns, ns) and (nt, nt) distance matrices.
    n_landmarks : int
        Landmark count for ``distance_mode="landmark"``.
    fgw_alpha : float
        FGW blending: 0 = pure GW, 1 = pure Wasserstein.
    C_linear : ndarray or Tensor, optional
        (ns, nt) feature cost for FGW.
    s_shared, M, alpha, max_iter, tol, epsilon, k : solver parameters
    min_iter_before_converge : int
    device : torch.device, optional
    verbose, verbose_every : progress printing
    log : bool
        Return (T, log_dict) if True.
    differentiable : bool
        Keep computation graph for backprop.
    grad_mode : str
        Gradient computation mode when ``differentiable=True``.
        ``"implicit"`` (default): exact via adjoint at Sinkhorn fixed point.
        Memory-efficient: O(NK).
        ``"unrolled"``: exact via unrolled PyTorch autograd.
        Memory: O(NK * sinkhorn_iters).
    semi_relaxed : bool
        Relax target marginal via KL penalty.
    rho_a, rho_b : float
        KL penalty weights for source and target marginals (semi_relaxed only).
    multiscale : bool
        Two-stage coarse-to-fine warm start.
    n_coarse : int, optional
        Coarse problem size (auto if None).
    lambda_ema_beta : float, optional
        EMA smoothing factor for the cost matrix. When set, maintains a
        running average: Lambda_ema = (1-beta)*Lambda_ema + beta*Lambda_sample.
        Reduces sampling variance at the cost of small bias that vanishes
        at convergence. Typical values: 0.3–0.7. None disables (default).
    mixed_precision : bool
        Run Sinkhorn iterations in float32 for speed, cast result back to
        float64. Safe because all critical ops are in log domain where
        values are O(log N). Marginals and transport plan stay in float64.

    Returns
    -------
    T : Tensor (ns, nt)
    log_dict : dict (only if log=True)
    """
    (X_source, X_target, p, q, dist_source, dist_target, C_linear_t,
     N, K, provider, device) = _prepare_inputs(
        X_source, X_target, p, q, dist_source, dist_target, C_linear,
        distance_mode, fgw_alpha, k, n_landmarks, device,
    )

    # Marginals (float64)
    if p is not None:
        p_real = p.to(dtype=torch.float64, device=device)
    else:
        p_real = torch.ones(N, device=device, dtype=torch.float64) / N
    if q is not None:
        q_real = q.to(dtype=torch.float64, device=device)
    else:
        q_real = torch.ones(K, device=device, dtype=torch.float64) / K

    # User-supplied warm start takes precedence; else multiscale; else uniform.
    if T_init is not None:
        if isinstance(T_init, np.ndarray):
            T_init = torch.from_numpy(T_init)
        T_init = T_init.to(device=device, dtype=torch.float64)
    else:
        T_init = _maybe_multiscale(
            multiscale, n_coarse, X_source, X_target, N, K,
            dist_source, dist_target, C_linear_t, fgw_alpha,
            distance_mode, n_landmarks, device, p_real, q_real,
            solver_fn=sampled_gw,
            solver_kwargs=dict(
                s_shared=s_shared, M=M, alpha=alpha, max_iter=max_iter, tol=tol,
                epsilon=epsilon, k=k, min_iter_before_converge=min_iter_before_converge,
                verbose=False, log=False, differentiable=differentiable,
                semi_relaxed=semi_relaxed, rho_a=rho_a, rho_b=rho_b,
            ),
        )
        if T_init is None:
            T_init = torch.outer(p_real, q_real)

    # C_linear on device
    C_lin_device = C_linear_t.to(dtype=torch.float64, device=device) if C_linear_t is not None and fgw_alpha > 0 else None

    # Sinkhorn function
    if differentiable and fgw_alpha == 0.0:
        warnings.warn(
            "differentiable=True with fgw_alpha=0 (pure GW): gradients cannot "
            "flow because the GW cost matrix is built from precomputed graph "
            "distances that are not part of the computation graph. Set "
            "fgw_alpha > 0 with a differentiable C_linear to get useful gradients.",
            stacklevel=2,
        )
    ctx = torch.no_grad() if not differentiable else torch.enable_grad()
    if differentiable:
        _gm = grad_mode
        def sinkhorn_fn(a, b, C, reg, **kw):
            kw.pop('_inplace_C', None)
            return _sinkhorn_differentiable(C, a, b, reg, grad_mode=_gm, **kw)
    else:
        sinkhorn_fn = _sinkhorn_torch

    with ctx:
        T_out, err_list, n_iter, gw_cost_val = _gw_loop(
            N=N, K=K, provider=provider,
            p_real=p_real, q_real=q_real, T_init=T_init,
            sinkhorn_fn=sinkhorn_fn, use_augmented=True,
            s_shared=s_shared,
            fgw_alpha=fgw_alpha, C_lin_device=C_lin_device,
            M=M, alpha=alpha, max_iter=max_iter, tol=tol,
            epsilon=epsilon, min_iter_before_converge=min_iter_before_converge,
            device=device, verbose=verbose, verbose_every=verbose_every,
            semi_relaxed=semi_relaxed, rho_a=rho_a, rho_b=rho_b,
            differentiable=differentiable,
            lambda_ema_beta=lambda_ema_beta,
            mixed_precision=mixed_precision,
        )

    if log:
        return T_out, {"err_list": err_list, "n_iter": n_iter, "gw_cost": gw_cost_val}
    return T_out


# ── Public API: low-rank solver ─────────────────────────────────────────

def sampled_lowrank_gw(
    X_source: np.ndarray | torch.Tensor | None = None,
    X_target: np.ndarray | torch.Tensor | None = None,
    p: np.ndarray | torch.Tensor | None = None,
    q: np.ndarray | torch.Tensor | None = None,
    *,
    rank: int = 20,
    lr_max_iter: int = 5,
    lr_dykstra_max_iter: int = 50,
    distance_mode: str = "dijkstra",
    dist_source: np.ndarray | torch.Tensor | None = None,
    dist_target: np.ndarray | torch.Tensor | None = None,
    n_landmarks: int = 50,
    fgw_alpha: float = 0.0,
    C_linear: np.ndarray | torch.Tensor | None = None,
    s_shared: int | None = None,
    M: int = 50,
    alpha: float = 0.9,
    max_iter: int = 500,
    tol: float = 1e-5,
    epsilon: float = 0.001,
    k: int = 30,
    min_iter_before_converge: int = 50,
    device: torch.device | None = None,
    verbose: bool = False,
    verbose_every: int = 20,
    log: bool = False,
    semi_relaxed: bool = False,
    rho_a: float = 1.0,
    rho_b: float = 1.0,
    multiscale: bool = False,
    n_coarse: int | None = None,
    lambda_ema_beta: float | None = None,
    mixed_precision: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Sampled Gromov-Wasserstein with low-rank Sinkhorn.

    Uses the low-rank Sinkhorn factorization (Scetbon, Cuturi & Peyre 2021)
    to reduce memory from O(NK) to O((N+K)*rank) per Sinkhorn step.

    This is a **memory optimization** for large-scale problems (N, K > 50k).
    At smaller scales, ``sampled_gw`` with standard Sinkhorn is faster.

    Parameters
    ----------
    X_source, X_target : ndarray or Tensor, optional
    p, q : ndarray or Tensor, optional
    rank : int
        Nonneg. rank of the transport plan factorization.
    lr_max_iter : int
        Outer mirror descent iterations per Sinkhorn call.
    lr_dykstra_max_iter : int
        Inner Dykstra projection iterations per Sinkhorn call.
    distance_mode, dist_source, dist_target, n_landmarks : distance params
    fgw_alpha, C_linear : Fused GW params
    s_shared, M, alpha, max_iter, tol, epsilon, k : solver params
    min_iter_before_converge : int
    device : torch.device, optional
    verbose, verbose_every : progress printing
    log : bool
    semi_relaxed : bool
    rho_a, rho_b : float
        KL penalty weights for source and target marginals (semi_relaxed only).
    multiscale : bool
    n_coarse : int, optional
    lambda_ema_beta : float, optional
        EMA smoothing factor for the cost matrix (see ``sampled_gw``).
    mixed_precision : bool
        Run internal computations in float32 for speed (see ``sampled_gw``).

    Returns
    -------
    T : Tensor (ns, nt)
    log_dict : dict (only if log=True)
    """
    if semi_relaxed:
        raise ValueError("semi_relaxed is not supported for low-rank Sinkhorn")
    if rho_a != rho_b:
        raise NotImplementedError(
            "sampled_lowrank_gw does not yet support rho_a != rho_b "
            "(low-rank Dykstra requires symmetric KL). Use sampled_gw for "
            "fully-unbalanced (rho_a, rho_b) FGW."
        )

    from torchgw._lowrank import sinkhorn_lowrank

    (X_source, X_target, p, q, dist_source, dist_target, C_linear_t,
     N, K, provider, device) = _prepare_inputs(
        X_source, X_target, p, q, dist_source, dist_target, C_linear,
        distance_mode, fgw_alpha, k, n_landmarks, device,
    )

    # Marginals (float64)
    if p is not None:
        p_real = p.to(dtype=torch.float64, device=device)
    else:
        p_real = torch.ones(N, device=device, dtype=torch.float64) / N
    if q is not None:
        q_real = q.to(dtype=torch.float64, device=device)
    else:
        q_real = torch.ones(K, device=device, dtype=torch.float64) / K

    # Multiscale warm start
    T_init = _maybe_multiscale(
        multiscale, n_coarse, X_source, X_target, N, K,
        dist_source, dist_target, C_linear_t, fgw_alpha,
        distance_mode, n_landmarks, device, p_real, q_real,
        solver_fn=sampled_lowrank_gw,
        solver_kwargs=dict(
            rank=rank, lr_max_iter=lr_max_iter,
            lr_dykstra_max_iter=lr_dykstra_max_iter,
            s_shared=s_shared, M=M, alpha=alpha, max_iter=max_iter, tol=tol,
            epsilon=epsilon, k=k, min_iter_before_converge=min_iter_before_converge,
            verbose=False, log=False,
            semi_relaxed=semi_relaxed, rho_a=rho_a, rho_b=rho_b,
        ),
    )
    if T_init is None:
        T_init = torch.outer(p_real, q_real)

    # C_linear on device
    C_lin_device = C_linear_t.to(dtype=torch.float64, device=device) if C_linear_t is not None and fgw_alpha > 0 else None

    # Wrap sinkhorn_lowrank with fixed rank/iteration params
    def _lr_sinkhorn(a, b, C, reg, semi_relaxed=False, rho_a=1.0, rho_b=1.0, verbose=False,
                     log_u_init=None, log_v_init=None):
        return sinkhorn_lowrank(
            a, b, C, rank=rank, reg=reg,
            max_iter=lr_max_iter, dykstra_max_iter=lr_dykstra_max_iter,
        )

    with torch.no_grad():
        T_out, err_list, n_iter, gw_cost_val = _gw_loop(
            N=N, K=K, provider=provider,
            p_real=p_real, q_real=q_real, T_init=T_init,
            sinkhorn_fn=_lr_sinkhorn, use_augmented=False,
            s_shared=s_shared,
            fgw_alpha=fgw_alpha, C_lin_device=C_lin_device,
            M=M, alpha=alpha, max_iter=max_iter, tol=tol,
            epsilon=epsilon, min_iter_before_converge=min_iter_before_converge,
            device=device, verbose=verbose, verbose_every=verbose_every,
            semi_relaxed=False, rho_a=rho_a, rho_b=rho_b,
            lambda_ema_beta=lambda_ema_beta,
            mixed_precision=mixed_precision,
        )

    if log:
        return T_out, {"err_list": err_list, "n_iter": n_iter, "gw_cost": gw_cost_val}
    return T_out
