# TorchGW Optimization Log

Performance optimization history for the TorchGW sampled Gromov-Wasserstein solver.
All benchmarks on NVIDIA L40S, PyTorch 2.6, CUDA 12.4 unless noted.

## Summary

Cumulative speedup from the full optimization pass:

| Benchmark | Before | After | Speedup |
|-----------|--------|-------|---------|
| spiral 400×500 dijkstra | 4.23s | 1.11s | **3.8x** |
| spiral 400×500 precomputed | 1.40s | 0.46s | **3.0x** |
| spiral 400×500 landmark | 2.84s | 0.47s | **6.0x** |
| spiral 4000×5000 landmark | — | 1.04s | — |
| random 2000×2500 precomputed | 6.14s | 2.56s | **2.4x** |
| random 5000×6000 precomputed | ~20s | 11.06s | **1.8x** |

Quality unchanged throughout: |Spearman ρ| ≥ 0.998 on spiral→swiss-roll.

---

## Phase 1: GPU Sampling + Kernel Fusion Prep

**GPU sampling via `torch.multinomial`** (`_sampling.py`)

Replaced CPU numpy sampling with GPU-native `torch.multinomial`. Previously
each GW iteration transferred the full N×K transport plan from GPU to CPU
(`.cpu().numpy()`). Now only 2×M integers are transferred back.

- Transfer reduction: O(NK) float64 → O(2M) int64 per iteration
- Measured speedup: **1.68x** on 2000×2500 precomputed

**Sinkhorn convergence check via logsumexp** (`_solver.py`)

The convergence check previously materialized the full N×K transport plan
just to compute row marginals. Replaced with:
```python
log_marginal = log_u + torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
```
Reduces memory from O(NK) to O(N) per check.

**tau=1.0 fast path** (`_solver.py`)

In balanced mode (default), `log_v = tau * log_v_raw + (1-tau) * log_v`
with tau=1.0 is a no-op multiply+add. Added branch to skip it.

**Pre-allocate Lambda_aug** (`_solver.py`)

Moved the `torch.zeros(N+1, K+1)` allocation outside the GW loop.
Eliminates one O(NK) allocation per iteration.

**LandmarkProvider device caching** (`_distances.py`)

Same pattern as PrecomputedProvider: cache the `.to(device)` result to avoid
repeated CPU→GPU transfers of the landmark embedding matrices.

**sample_pairs returns arrays** (`_sampling.py`)

Changed return type from `list[tuple[int, int]]` to `(ndarray, ndarray)`,
eliminating Python tuple construction and zip/unzip overhead.

---

## Phase 2: Dijkstra Caching

**Per-node Dijkstra cache in DijkstraProvider** (`_distances.py`)

High-weight nodes are repeatedly sampled as anchors across GW iterations.
Added a dict cache mapping `node_id → distance_row` with FIFO eviction
at 2000 rows per side.

- 500×600, 50 iters: 0.96s → 0.60s (**1.6x**)
- 1000×1200, 100 iters: 3.08s → 1.40s (**2.2x**)
- 2000×2500, 100 iters: 6.13s → 3.41s (**1.8x**)

---

## Phase 3: Mixed Precision

**float32 Sinkhorn + float64 marginals** (`_solver.py`)

All log-domain Sinkhorn values are O(log N) magnitude, safe in float32.
When `mixed_precision=True`:
- `sink_dtype = float32` for the entire GW loop (T_real, Lambda_aug, Sinkhorn)
- Only the final output is cast back to float64
- Zero per-iteration dtype conversions (marginals pre-cast outside loop)

Quality impact: none (|ρ| 0.9994 → 0.9993 on spiral benchmark).

- spiral 400×500: 3.29s → 1.88s (**1.7x**)
- 2000×2500 precomputed: 3.72s → 3.08s (**1.2x**)

Note: L40S has FP64 = FP32/2. On consumer GPUs (FP64 = FP32/64), the
speedup would be much larger.

---

## Phase 4: Early Stopping

**GW cost plateau detection** (`_solver.py`)

The existing convergence criterion `err = ||T - T_prev|| < tol` measures
sampling noise, not optimization progress, and rarely triggers. Added
cost plateau detection via EMA:

- Track EMA of GW cost (alpha=0.2)
- If EMA doesn't improve by >0.5% for `patience` consecutive iterations, stop
- `patience = max(min_iter_before_converge // 2, 20)`

Correctly stops early when cost stabilizes (dijkstra mode: 500 → 97 iters)
while NOT stopping when cost is still improving (spiral with reg annealing).

---

## Phase 5: Sync Reduction

**Fewer CUDA synchronization points** (`_solver.py`)

- Replaced `Lambda.max().item()` (GPU→CPU sync) with `Lambda.max().clamp(min=1.0)` (stays on GPU)
- Batch convergence checks: compute GW cost + err with `.item()` only every 5 iterations
- Hoisted marginal dtype casts outside the loop
- Removed periodic `gc.collect() + cuda.empty_cache()` (added Python overhead without memory benefit)

---

## Phase 6: Triton Fused Sinkhorn

**Custom Triton kernels** (`_triton_sinkhorn.py`)

Three fused kernels replacing PyTorch multi-kernel sequences:

1. **Row update kernel**: `log_u[n] = log_a[n] - logsumexp_k(log_K[n,k] + log_v[k])`
   - Single-pass online logsumexp (amax + sub + exp + sum + log fused into 1 kernel)
   - No intermediate N×K matrix
   - Tiles over K in configurable BLOCK_K chunks

2. **Column update kernel**: same structure, tiles over N

3. **Marginal error kernel**: fused convergence check via `atomic_max(|marginal - a|)`,
   avoids materializing N×K for the check

4. **T materialization kernel**: `T[n,k] = exp(log_u[n] + log_K[n,k] + log_v[k])`
   written directly, no intermediate broadcast

Dispatch: Triton (CUDA) → torch.compile (CUDA) → pure PyTorch (CPU/fallback).
No new dependencies (Triton ships with PyTorch 2.0+).

Forced 100 iterations benchmark:

| Matrix | dtype | PyTorch | Triton | Speedup |
|--------|-------|---------|--------|---------|
| 5001×6001 | fp32 | 261ms | 50ms | **5.2x** |
| 10001×12001 | fp32 | 1175ms | 257ms | **4.6x** |
| 10001×12001 | fp64 | 2433ms | 1064ms | **2.3x** |

---

## Phase 7: Sinkhorn Warm-Start

**Reuse potentials across GW iterations** (`_solver.py`)

Adjacent GW iterations have similar cost matrices (only anchor sampling
changes). The previous Sinkhorn solution is a good starting point for the
next. Store `log_u`/`log_v` on the returned tensor and pass them as
`log_u_init`/`log_v_init` to the next Sinkhorn call.

Reduces Sinkhorn convergence from ~10 to ~3-5 iterations per GW step.
Supported in all three backends (Triton, torch.compile, PyTorch fallback).

---

## Phase 8: Parallel Preprocessing

**Process-parallel all-pairs Dijkstra** (`_distances.py`)

`PrecomputedProvider` runs all-pairs Dijkstra on source and target graphs.
These are independent and can run in parallel. Since scipy's Dijkstra
holds the GIL, thread parallelism doesn't help — use `joblib` process
parallelism instead. Only activates when total nodes ≥ 2000 (below that,
process spawn overhead exceeds savings).

- 2000×2500: 2.52s → 2.16s (**1.2x**)
- 5000×6000: 16.24s → 10.90s (**1.5x**)

---

## Bug Fixes During Optimization

Fixes applied during the optimization process:

- `torch.log(a + 1e-300)` → `.clamp(min=1e-30)` (1e-300 vanishes in float32)
- kNN graph symmetrization via `.maximum(.T)`
- Semi-relaxed Sinkhorn tau damping: proper KL proximal blend
- `differentiable=True` + `semi_relaxed=True`: raise NotImplementedError
- Regularization decay capped at 10x
- Handle all-inf distance matrices
- Low-rank mirror descent gamma lower bound
- Detach T_prev in differentiable mode to prevent graph accumulation
- SVD dimension bounds in joint_embedding
- scipy.sparse.linalg.cg tol/rtol compatibility
- DijkstraProvider cache eviction safety (don't evict needed keys)
- `sample_pairs_gpu` cast to float32 for `torch.multinomial`
- `lambda_ema_beta=0.0` treated as disabled
- `sampled_lowrank_gw` semi_relaxed check moved to function start

---

## Remaining Bottlenecks

| Bottleneck | Impact | Mitigation |
|-----------|--------|------------|
| All-pairs Dijkstra (precomputed mode) | 90%+ of wall clock at large scale | Use `distance_mode="landmark"` |
| Per-iteration Dijkstra (dijkstra mode) | Scales with unique nodes sampled | Cache + early stopping |
| Python loop overhead | ~1-2ms/iter at small scale | Requires torch.compile (unavailable on some systems) |
| scipy Dijkstra holds GIL | Process parallelism has pickle overhead | Waiting for free-threaded Python (3.13+) or cuGraph |

## Future Directions

See `docs/improvements.md` for torch.compile, Triton extensions, cuGraph,
and algorithmic improvements.
