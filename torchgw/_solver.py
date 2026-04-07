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
    log_K: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    tau: float,
    max_iter: int,
    tol: float,
    check_every: int,
    a: torch.Tensor,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run log-domain Sinkhorn iterations. Returns (log_u, log_v).

    Uses torch.compile for kernel fusion when running on CUDA.
    Runs check_every iterations as a compiled batch, then checks convergence.
    """
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    is_balanced = (tau == 1.0)

    # Use compiled kernel on CUDA when not verbose and convergence check is needed
    use_compiled = (
        log_K.is_cuda
        and not verbose
        and check_every > 1
        and tol > 0
    )
    if use_compiled:
        try:
            iter_fn = _get_compiled_sinkhorn()
        except Exception:
            use_compiled = False

    done = 0
    while done < max_iter:
        if use_compiled and not verbose:
            batch = min(check_every, max_iter - done)
            log_u, log_v = iter_fn(
                log_K, log_a, log_b, log_u, log_v,
                is_balanced, tau, batch,
            )
            done += batch
        else:
            # Fallback: single step
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_v_raw = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
            if is_balanced:
                log_v = log_v_raw
            else:
                log_v = tau * log_v_raw + (1 - tau) * log_v
            done += 1

        # Convergence check
        if tol > 0 and done % check_every == 0:
            log_marginal = log_u + torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            marginal_err = torch.abs(torch.exp(log_marginal) - a).max().item()
            if verbose:
                print(f"    sinkhorn {done:>4}/{max_iter} | marginal_err: {marginal_err:.4e}")
            if marginal_err < tol:
                if verbose:
                    print(f"    sinkhorn converged at {done} (err={marginal_err:.4e})")
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
    rho: float = 1.0,
    verbose: bool = False,
) -> torch.Tensor:
    """Log-domain Sinkhorn for numerical stability. Pure PyTorch.

    Operates in whatever dtype the inputs are given (float32 or float64).
    Dtype selection is handled by the caller (_gw_loop via sink_dtype).
    """
    log_K = -C / reg
    log_a = torch.log(a.clamp(min=1e-30))
    log_b = torch.log(b.clamp(min=1e-30))
    tau = rho / (rho + reg) if semi_relaxed else 1.0

    log_u, log_v = _sinkhorn_loop(log_K, log_a, log_b, tau, max_iter, tol, check_every, a, verbose=verbose)
    return torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))


class _SinkhornAutograd(torch.autograd.Function):
    """Memory-efficient differentiable Sinkhorn.

    Gradient via envelope theorem: dL/dC = -T * grad_T / reg.
    """

    @staticmethod
    def forward(ctx, C, a, b, reg, max_iter, tol, check_every, semi_relaxed, rho):
        log_K = -C / reg
        log_a = torch.log(a.clamp(min=1e-30))
        log_b = torch.log(b.clamp(min=1e-30))
        tau = rho / (rho + reg) if semi_relaxed else 1.0

        log_u, log_v = _sinkhorn_loop(log_K, log_a, log_b, tau, max_iter, tol, check_every, a)
        T = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))

        ctx.save_for_backward(T)
        ctx.reg = reg
        return T

    @staticmethod
    def backward(ctx, grad_T):
        (T,) = ctx.saved_tensors
        grad_C = -grad_T * T / ctx.reg
        return grad_C, None, None, None, None, None, None, None, None


def _sinkhorn_differentiable(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    reg: float,
    max_iter: int = 100,
    tol: float = 5e-4,
    check_every: int = 10,
    semi_relaxed: bool = False,
    rho: float = 1.0,
    verbose: bool = False,
) -> torch.Tensor:
    """Differentiable Sinkhorn using custom autograd (memory-efficient)."""
    if semi_relaxed:
        raise NotImplementedError(
            "differentiable=True is not supported with semi_relaxed=True: "
            "the envelope theorem gradient is only valid for balanced Sinkhorn"
        )
    return _SinkhornAutograd.apply(
        C, a, b, reg, max_iter, tol, check_every, semi_relaxed, rho,
    )


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
    rho: float,
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

    # Pre-allocate augmented cost matrix and cast marginals (reused every iteration)
    if use_augmented:
        Lambda_aug = torch.zeros(N + 1, K + 1, device=device, dtype=sink_dtype)
        p_aug_sink = p_aug.to(sink_dtype)
        q_aug_sink = q_aug.to(sink_dtype)
    p_sink = p_real.to(sink_dtype)
    q_sink = q_real.to(sink_dtype)

    for i in range(max_iter):
        current_reg = initial_reg * (decay ** i)
        # In differentiable mode, detach T_prev to prevent computation graph
        # accumulation across iterations; only the final iteration's Sinkhorn
        # step will carry gradients through the momentum blend.
        T_prev = T_real.detach().clone() if differentiable else T_real.clone()

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

            term_A = torch.mean(D_left ** 2, dim=1, keepdim=True)
            term_C = torch.mean(D_tgt ** 2, dim=1, keepdim=True).T
            term_B = -2 * (D_left @ D_tgt.T) / M
            Lambda_gw = term_A + term_B + term_C

            # Lambda EMA: smooth cost matrix across iterations
            if lambda_ema_beta is not None:
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
            max_val = Lambda.max().item()
            penalty = 100.0 * max_val if max_val > 0 else 100.0
            Lambda_aug[:-1, -1] = penalty
            Lambda_aug[-1, :-1] = penalty
            Lambda_aug[-1, -1] = 0.0

            verbose_sink = verbose and (n_iter + 1) % verbose_every == 0
            T_aug = sinkhorn_fn(p_aug_sink, q_aug_sink,
                                Lambda_aug, current_reg,
                                semi_relaxed=semi_relaxed, rho=rho,
                                verbose=verbose_sink)
            T_new = T_aug[:-1, :-1]
        else:
            verbose_sink = verbose and (n_iter + 1) % verbose_every == 0
            Lambda_sink = Lambda if Lambda.dtype == sink_dtype else Lambda.to(sink_dtype)
            T_new = sinkhorn_fn(p_sink, q_sink,
                                Lambda_sink,
                                current_reg, semi_relaxed=semi_relaxed, rho=rho,
                                verbose=verbose_sink)

        # Momentum update
        T_real = (1 - alpha) * T_prev + alpha * T_new

        # GW cost (unregularized) — dot product avoids N×K intermediate
        Lambda_flat = Lambda.reshape(-1) if Lambda.dtype == sink_dtype else Lambda.to(sink_dtype).reshape(-1)
        gw_cost_val = torch.dot(Lambda_flat, T_real.reshape(-1)).item()

        err = torch.linalg.norm(T_real - T_prev).item()
        err_list.append(err)
        n_iter = i + 1
        if verbose and (n_iter % verbose_every == 0 or i == max_iter - 1):
            print(f"  iter {n_iter:>4}/{max_iter} | err: {err:.4e} | "
                  f"gw_cost: {gw_cost_val:.4e} | reg: {current_reg:.4e}")

        if err < tol and i >= min_iter_before_converge:
            if verbose:
                print(f"  converged at iteration {n_iter} (err={err:.4e})")
            break

        del T_new
        if use_augmented:
            del T_aug
        if provider is not None:
            del D_left, D_tgt, Lambda_gw

    # Cast back to float64 for output precision
    T_out = T_real if T_real.dtype == torch.float64 else T_real.to(torch.float64)
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
    semi_relaxed: bool = False,
    rho: float = 1.0,
    multiscale: bool = False,
    n_coarse: int | None = None,
    lambda_ema_beta: float | None = None,
    mixed_precision: bool = False,
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
    semi_relaxed : bool
        Relax target marginal via KL penalty.
    rho : float
        KL penalty weight (semi_relaxed only).
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

    # Multiscale warm start
    T_init = _maybe_multiscale(
        multiscale, n_coarse, X_source, X_target, N, K,
        dist_source, dist_target, C_linear_t, fgw_alpha,
        distance_mode, n_landmarks, device, p_real, q_real,
        solver_fn=sampled_gw,
        solver_kwargs=dict(
            s_shared=s_shared, M=M, alpha=alpha, max_iter=max_iter, tol=tol,
            epsilon=epsilon, k=k, min_iter_before_converge=min_iter_before_converge,
            verbose=False, log=False, differentiable=differentiable,
            semi_relaxed=semi_relaxed, rho=rho,
        ),
    )
    if T_init is None:
        T_init = torch.outer(p_real, q_real)

    # C_linear on device
    C_lin_device = C_linear_t.float().to(device) if C_linear_t is not None and fgw_alpha > 0 else None

    # Sinkhorn function
    ctx = torch.no_grad() if not differentiable else torch.enable_grad()
    sinkhorn_fn = _sinkhorn_differentiable if differentiable else _sinkhorn_torch

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
            semi_relaxed=semi_relaxed, rho=rho,
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
    rho: float = 1.0,
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
    rho : float
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
            semi_relaxed=semi_relaxed, rho=rho,
        ),
    )
    if T_init is None:
        T_init = torch.outer(p_real, q_real)

    # C_linear on device
    C_lin_device = C_linear_t.float().to(device) if C_linear_t is not None and fgw_alpha > 0 else None

    # Wrap sinkhorn_lowrank with fixed rank/iteration params
    def _lr_sinkhorn(a, b, C, reg, semi_relaxed=False, rho=1.0, verbose=False):
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
            semi_relaxed=False, rho=rho,
            lambda_ema_beta=lambda_ema_beta,
            mixed_precision=mixed_precision,
        )

    if log:
        return T_out, {"err_list": err_list, "n_iter": n_iter, "gw_cost": gw_cost_val}
    return T_out
