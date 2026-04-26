# Changelog

All notable changes to TorchGW are documented in this file.

## [Unreleased]

### Added

- `sampled_gw(rho_a, rho_b, semi_relaxed=True)` — fully-unbalanced FGW with
  two-sided KL marginal damping. PyTorch fallback path only; Triton fast-
  path falls back to PyTorch when `rho_a != rho_b`. Lowrank
  (`sampled_lowrank_gw`) raises `NotImplementedError` when `rho_a != rho_b`.
- Cross-validation test against POT's `sinkhorn_unbalanced` to verify
  numerical correctness of the new path.

### Changed

- The `rho` kwarg on `sampled_gw` and `sampled_lowrank_gw` is preserved
  as a backward-compatible alias for legacy single-side semi-relaxed
  semantics: `rho=X` is equivalent to `rho_a=1.0, rho_b=X` (damping only
  on the v/target marginal — exactly the pre-PR behavior). New callers
  should prefer the explicit `(rho_a, rho_b)` form.

---

## [0.4.1] — 2026-04-09

Exact differentiable gradients via implicit differentiation. Fixes a
correctness bug where the previous "envelope theorem" backward produced
gradients with up to 30x error (cosine similarity as low as 0.07).

### Breaking Changes

- Default gradient computation for `differentiable=True` is now **implicit
  differentiation** (exact) instead of the old frozen-potentials approximation.
  No API change needed — the new default is strictly better.

### New Features

- **`grad_mode` parameter** for `sampled_gw` — controls how gradients are
  computed when `differentiable=True`:
  - `"implicit"` (default): exact gradient via adjoint system at the Sinkhorn
    fixed point. Solved via Schur complement on the Sinkhorn Jacobian.
    Memory: O(NK + K^2). Same speed as the old approximate mode.
  - `"unrolled"`: exact gradient via unrolled PyTorch autograd. Memory:
    O(NK * sinkhorn_iters). Useful as fallback at extremely small epsilon.

### Bug Fixes

- **Gradient correctness** — The old backward formula `grad_C = -grad_T * T / reg`
  treated Sinkhorn potentials as constants ("frozen-potentials"). This is a first-order
  approximation that ignores how the potentials depend on C through the Sinkhorn
  iterations. The new implicit differentiation backward solves the adjoint system
  derived from the implicit function theorem at the Sinkhorn fixed point, giving
  exact gradients.

- **Adjoint solver stability** — Initial implementation used fixed-point iteration
  for the adjoint system, which diverges when the spectral radius >= 1 (common at
  small epsilon). Replaced with Schur complement direct solve on the Sinkhorn
  Jacobian J^T (eigenvalues in [0,2], well-conditioned). Null space from potential
  constant ambiguity removed via rank-1 correction (11^T/K).

- **Warning for non-differentiable pure GW** — `differentiable=True` with
  `fgw_alpha=0` now emits a warning, since gradients cannot flow through
  precomputed graph distances.

### Internal

- `_SinkhornAutograd` renamed to `_SinkhornApproximate` (frozen-potentials,
  used internally for semi-relaxed only)
- New `_SinkhornImplicit` class (exact VJP via adjoint)
- New `_sinkhorn_unrolled` function (exact VJP via autograd)
- `_sinkhorn_differentiable` rewritten as dispatcher for the three backends

### Tests

- 8 new gradient correctness tests in `tests/test_sinkhorn_grad.py`:
  implicit vs unrolled (rel_err < 2%), implicit vs finite differences,
  descent direction, non-uniform marginals, approximate formula verification,
  grad_mode validation
- Integration tests for `grad_mode` through `sampled_gw`
- 111 total tests, all passing

### Documentation

- `docs/algorithm.rst`: full derivation of implicit differentiation
  (Jacobian structure, adjoint equation, Schur complement, null-space handling)
- README: updated news, differentiable mode example with `grad_mode`

---

## [0.4.0] — 2026-04-07

Major performance and robustness release. **3-6x faster** on typical workloads,
with Triton GPU kernel acceleration, mixed precision support, and comprehensive
numerical stability fixes.

### Performance

- **Triton fused Sinkhorn kernels** — Custom GPU kernels for the Sinkhorn row/column
  logsumexp updates, reducing kernel launches from ~6 to 1 per half-step. **2-5x speedup**
  on the Sinkhorn portion (5001×6001 fp32: 261ms → 50ms). Includes fused transport plan
  materialization and fused marginal error check. Falls back to PyTorch automatically
  when Triton is unavailable. (`_triton_sinkhorn.py`)
- **Sinkhorn warm-start** — Reuse log-domain potentials (log_u, log_v) from the previous
  GW iteration as initial values. Reduces Sinkhorn convergence from ~10 to ~3-5 steps.
- **GPU sampling** — Replace CPU numpy sampling with `torch.multinomial` on GPU.
  Transfers 2×M integers instead of the full N×K transport plan per iteration.
- **Mixed precision** — New `mixed_precision=True` parameter runs Sinkhorn in float32
  (safe in log domain) while keeping marginals and output in float64. Up to 1.7x faster
  on A100/L40S; larger gains expected on consumer GPUs.
- **Dijkstra caching** — `DijkstraProvider` caches per-node SSSP results across iterations
  with FIFO eviction (max 2000 rows per side). Avoids redundant computation when the
  same anchor nodes are re-sampled.
- **Cost plateau early stopping** — GW cost EMA + patience-based convergence detection.
  Stops when the smoothed cost stops improving, rather than waiting for the noisy
  `||T - T_prev||` to drop below `tol` (which may never happen due to sampling noise).
  Example: dijkstra 1000×1200 stops at 97 iters instead of running all 500.
- **Parallel all-pairs Dijkstra** — `PrecomputedProvider` runs source and target
  graph Dijkstra in parallel via process-based parallelism (scipy holds the GIL).
  1.2-1.5x speedup on large graphs (≥2000 total nodes).
- **Reduced CUDA sync points** — Convergence checks batched every 5 iterations;
  augmented penalty computed on GPU without `.item()` sync.
- **Sinkhorn convergence check via logsumexp** — Avoids materializing full N×K matrix
  for the marginal error computation.
- **Pre-allocated augmented cost matrix** — Reused across iterations instead of
  re-allocated each step.

### New Features

- `mixed_precision` parameter for `sampled_gw` and `sampled_lowrank_gw`
- `lambda_ema_beta` parameter for cost matrix EMA smoothing (variance reduction)
- Verbose Sinkhorn output (`verbose=True` prints per-iteration marginal errors)
- `sample_pairs_gpu()` — GPU-native weighted sampling function
- `sample_pairs_from_plan()` now accepts optional `rng` parameter for reproducibility

### Bug Fixes

- **Numerical stability**
  - `torch.log(a + 1e-300)` replaced with `.clamp(min=1e-30)` — the 1e-300 constant
    vanishes in float32, providing no protection against log(0)
  - Regularization decay capped at 10x to prevent instability with large epsilon values
  - Low-rank mirror descent: enforce `gamma * reg >= 1` to prevent exponential overflow
  - Handle all-inf distance matrices (fully disconnected subgraphs) without crashing
  - `sample_pairs_gpu` casts to float32 before `torch.multinomial` (required on some
    PyTorch versions/devices)

- **Correctness**
  - kNN graph symmetrized via `.maximum(.T)` — `kneighbors_graph` returns directed edges
  - Semi-relaxed Sinkhorn: correct KL proximal blend `tau * new + (1-tau) * old` instead
    of `tau * new` which discarded history
  - `differentiable=True` + `semi_relaxed=True` now raises `NotImplementedError`
    (envelope theorem gradient is invalid for unbalanced Sinkhorn)
  - Detach `T_prev` in differentiable mode to prevent computation graph accumulation
    across GW iterations (OOM after many iterations)
  - `joint_embedding`: prevent index out-of-bounds when `out_dim > k_svds`
  - `lambda_ema_beta=0.0` now disables EMA (previously locked to first iteration's cost)
  - Dijkstra cache eviction safety: never evict keys needed by the current request

- **Compatibility**
  - `scipy.sparse.linalg.cg`: auto-detect `tol` vs `rtol` parameter name for
    SciPy 1.10-1.17+ compatibility
  - `sampled_lowrank_gw`: `semi_relaxed` validation moved to function start (fail-fast)

- **API consistency**
  - `sampled_lowrank_gw` now accepts `mixed_precision` parameter
  - Removed unused `semi_relaxed`/`rho`/`**kwargs` from `sinkhorn_lowrank` signature
  - `sample_pairs_from_plan` returns `(rows, cols)` arrays instead of `list[tuple]`

### Tests

- 72 tests covering all solver modes, mixed precision, early stopping, Dijkstra
  cache, differentiable gradients, boundary values, and semi-relaxed mode
- Test suite runs in ~18s (down from ~68s before optimizations)

### Documentation

- `docs/optimization-log.md` — Detailed optimization history with benchmarks
- `docs/improvements.md` — Updated future directions (torch.compile, cuGraph, Triton extensions)

---

## [0.3.0] — 2026-04-03

Initial public release.

- Sampled Gromov-Wasserstein solver (`sampled_gw`) with log-domain Sinkhorn
- Low-rank solver (`sampled_lowrank_gw`) via mirror descent + Dykstra
- Three distance modes: `dijkstra`, `precomputed`, `landmark`
- Fused Gromov-Wasserstein (`fgw_alpha` blending)
- Multiscale warm-start via farthest-point sampling
- Differentiable transport plans (`differentiable=True`)
- Semi-relaxed mode for unbalanced transport
- Joint manifold embedding (`joint_embedding`)
- kNN graph construction with component stitching (`build_knn_graph`)
