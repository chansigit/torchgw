<p align="center">
  <img src="docs/logo.svg" alt="TorchGW logo" width="620">
</p>

# TorchGW — Fast Sampled Gromov-Wasserstein

[![Docs](https://img.shields.io/badge/docs-chansigit.github.io%2Ftorchgw-blue)](https://chansigit.github.io/torchgw/)
[![GitHub](https://img.shields.io/badge/github-chansigit%2Ftorchgw-black?logo=github)](https://github.com/chansigit/torchgw)
[![Version](https://img.shields.io/badge/version-0.4.0-green)](CHANGELOG.md)

A **pure PyTorch** solver for [Gromov-Wasserstein](https://arxiv.org/abs/1805.09114) optimal transport.
Aligns two point clouds by matching their internal distance structures — even when the point clouds
live in different dimensions.

**Core idea:** instead of the full *O(NK(N+K))* GW cost, TorchGW samples *M* anchor pairs each iteration
and approximates the cost in *O(NKM)*, enabling GPU-accelerated alignment at scales where
standard solvers are impractical.

---

## News

**v0.4.0** (2026-04-07) — Performance & robustness release

- **Triton fused Sinkhorn kernels** — 2-5x speedup on GPU via single-pass online logsumexp, no N×K intermediate matrices
- **Mixed precision** — `mixed_precision=True` runs Sinkhorn in float32, output stays float64
- **Smart early stopping** — cost plateau detection stops when the solver has converged, not when it hits `max_iter`
- **Sinkhorn warm-start** — reuses potentials across GW iterations, halving Sinkhorn steps
- **Dijkstra caching** — avoids redundant shortest-path computations across iterations
- **15 numerical stability & correctness fixes** — log-domain safety, kNN symmetrization, semi-relaxed damping, and more
- See [CHANGELOG.md](CHANGELOG.md) for the full list

---

## Installation

```bash
pip install -e .
```

Requires `numpy`, `scipy`, `scikit-learn`, `torch`, `joblib`. No POT at runtime.
Triton (ships with PyTorch 2.0+) enables GPU kernel fusion automatically.

## Quick start

```python
import torch
from torchgw import sampled_gw

X = torch.randn(500, 3)   # source (500 points, 3D)
Y = torch.randn(600, 5)   # target (600 points, 5D — dimensions may differ)

T = sampled_gw(X, Y, epsilon=0.005, M=80, max_iter=200)
# T[i,j] = coupling weight between X[i] and Y[j]
```

For best GPU performance on large problems:

```python
T = sampled_gw(X, Y, distance_mode="landmark", mixed_precision=True)
```

## How it works

Each iteration:

1. **Sample** *M* anchor pairs from the current transport plan *T* (GPU multinomial)
2. **Compute distances** from all points to the sampled anchors (Dijkstra / precomputed / landmark)
3. **Assemble GW cost matrix** from the sampled distances
4. **Sinkhorn projection** to obtain a new transport plan (Triton fused log-domain kernels)
5. **Momentum update** with warm-started potentials

Key design choices:
- Triton fused Sinkhorn on GPU — single-pass online logsumexp, no intermediate N×K matrices
- Automatic fallback to pure PyTorch when Triton is unavailable (CPU, older PyTorch)
- Mixed precision: float32 Sinkhorn + float64 marginals/output for speed without quality loss
- Smart convergence: cost plateau detection + transport plan norm check

## Benchmark

Spiral (2D) to Swiss roll (3D), NVIDIA L40S:

| Scale | Method | Time | Spearman |
|-------|--------|:----:|:--------:|
| 400 vs 500 | POT | 1.6s | 0.999 |
| 400 vs 500 | **TorchGW** | **0.46s** | **0.999** |
| 4000 vs 5000 | POT | 183s | 0.999 |
| 4000 vs 5000 | **TorchGW** (precomputed) | **5.07s** | **0.998** |
| 4000 vs 5000 | **TorchGW** (landmark) | **1.04s** | **0.999** |

At 4000×5000 with landmark distances, **TorchGW is ~175x faster** than POT.

<details>
<summary>Benchmark plots (click to expand)</summary>

### 400 vs 500
![400 vs 500](docs/demo_spiral_to_swissroll_400v500.png)

### 4000 vs 5000
![4000 vs 5000](docs/demo_spiral_to_swissroll_4000v5000.png)

</details>

**Distance mode scaling guide** — choose based on your data size:

| Mode | Best for | Per-iteration cost | Memory |
|------|:--------:|:------------------:|:------:|
| `"precomputed"` | N < 5k | O(NM) lookup | O(N²) |
| `"dijkstra"` (default) | 5k–50k | O(MN log N) | O(NM) |
| `"landmark"` | **N > 5k recommended** | O(NMd) GPU | O(Nd) |

> For most use cases, `distance_mode="landmark"` gives the best speed/quality tradeoff.

---

## API Reference

### `sampled_gw` — standard solver

```python
sampled_gw(
    X_source=None,            # (N, D) features, Tensor or ndarray
    X_target=None,            # (K, D') features
    p=None, q=None,           # marginals (uniform if None)
    *,
    distance_mode="dijkstra", # "precomputed" | "dijkstra" | "landmark"
    dist_source=None,         # (N, N) precomputed distance matrix
    dist_target=None,         # (K, K) precomputed distance matrix
    n_landmarks=50,           # for distance_mode="landmark"
    fgw_alpha=0.0,            # 0=GW, 1=Wasserstein, between=Fused GW
    C_linear=None,            # (N, K) feature cost for FGW
    M=50,                     # anchor pairs per iteration
    alpha=0.9,                # momentum
    max_iter=500, tol=1e-5,   # convergence
    epsilon=0.001,            # entropic regularization
    k=30,                     # kNN neighbors
    semi_relaxed=False,       # relax target marginal
    rho=1.0,                  # KL penalty (semi-relaxed only)
    differentiable=False,     # keep autograd graph
    multiscale=False,         # coarse-to-fine warm start
    n_coarse=None,            # coarse size (auto if None)
    lambda_ema_beta=None,     # EMA smoothing for cost matrix (0–1)
    mixed_precision=False,    # float32 Sinkhorn for speed
    device=None, verbose=False, log=False,
) -> Tensor                   # (N, K) transport plan
```

### `sampled_lowrank_gw` — memory-optimized solver

Same interface as `sampled_gw`, plus low-rank parameters.
Uses [Scetbon, Cuturi & Peyre (2021)](https://arxiv.org/abs/2103.04737) factorization
to reduce Sinkhorn memory from O(NK) to O((N+K)r).

```python
sampled_lowrank_gw(
    ...,                       # same as sampled_gw
    rank=20,                   # transport plan rank
    lr_max_iter=5,             # mirror descent iterations
    lr_dykstra_max_iter=50,    # Dykstra projection iterations
) -> Tensor
```

> **When to use:** only when N*K is too large for standard Sinkhorn memory (N, K > 50k).
> At smaller scales, `sampled_gw` is significantly faster.

---

## Usage Guide

### Distance strategies

```python
# Small scale (N < 5k): precompute all-pairs distances once
T = sampled_gw(X, Y, distance_mode="precomputed")

# Or pass your own distance matrices (skips graph construction)
T = sampled_gw(dist_source=D_X, dist_target=D_Y, distance_mode="precomputed")

# Large scale (recommended): landmark Dijkstra (FPS + GPU cdist)
T = sampled_gw(X, Y, distance_mode="landmark", n_landmarks=50)
```

### Best performance settings

```python
T = sampled_gw(
    X, Y,
    distance_mode="landmark",   # avoids expensive all-pairs Dijkstra
    mixed_precision=True,       # float32 Sinkhorn (2x faster on GPU)
    M=80,                       # more samples = better cost estimate
    epsilon=0.005,              # moderate regularization
)
```

### Fused Gromov-Wasserstein

Blend structural (graph distance) and feature (linear) costs:

```python
C_feat = torch.cdist(features_src, features_tgt)
T = sampled_gw(X, Y, fgw_alpha=0.5, C_linear=C_feat)

# Pure Wasserstein (no graph distances needed)
T = sampled_gw(fgw_alpha=1.0, C_linear=C_feat)
```

### Semi-relaxed GW

For unbalanced datasets (e.g., cell types present in one but not the other):

```python
T = sampled_gw(X, Y, semi_relaxed=True, rho=1.0)
# Source marginal enforced, target marginal soft (KL penalty weighted by rho)
```

### Multi-scale warm start

Speeds up convergence by solving a coarse problem first:

```python
T = sampled_gw(X, Y, multiscale=True)
T = sampled_gw(X, Y, multiscale=True, n_coarse=200)
```

> GW has symmetric local optima. The coarse solve may find a different optimum than
> the fine solve would. Works best on data without strong symmetries.

### Differentiable mode

Use GW cost as a training loss (gradients flow via envelope theorem):

```python
C_feat = torch.cdist(encoder(X), encoder(Y))  # from a learnable encoder
T = sampled_gw(fgw_alpha=1.0, C_linear=C_feat, differentiable=True)
loss = (C_feat.detach() * T).sum()
loss.backward()  # gradients flow to encoder parameters
```

### Low-rank Sinkhorn

For very large problems where the N*K transport plan does not fit in memory:

```python
from torchgw import sampled_lowrank_gw

T = sampled_lowrank_gw(X, Y, rank=30, distance_mode="landmark", n_landmarks=50)
```

---

## Examples

```bash
python examples/benchmark_distance_modes.py       # distance mode comparison
pip install pot && python examples/demo_spiral_to_swissroll.py  # TorchGW vs POT
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
