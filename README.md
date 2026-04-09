<p align="center">
  <img src="docs/logo.svg" alt="TorchGW logo" width="620">
</p>

<h3 align="center">Fast Sampled Gromov-Wasserstein Optimal Transport</h3>

<p align="center">
  <a href="https://chansigit.github.io/torchgw/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=readthedocs&logoColor=white" alt="Documentation"></a>
  <a href="https://github.com/chansigit/torchgw"><img src="https://img.shields.io/badge/github-chansigit%2Ftorchgw-black?logo=github" alt="GitHub"></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.4.0-green" alt="Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Non--Commercial-orange" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
</p>

<p align="center">
  <b>Pure PyTorch</b> &nbsp;|&nbsp; <b>Triton GPU Kernels</b> &nbsp;|&nbsp; <b>Differentiable</b> &nbsp;|&nbsp; <b>Up to 175x faster than POT</b>
</p>

---

TorchGW aligns two point clouds by matching their internal distance structures --
even when they live in **different dimensions**. Instead of the full *O(NK(N+K))* GW cost,
it samples *M* anchor pairs each iteration and approximates the cost in *O(NKM)*,
enabling GPU-accelerated alignment at scales where standard solvers are impractical.

**Use cases:** single-cell multi-omics integration, cross-domain graph matching,
shape correspondence, manifold alignment.

## Highlights

<table>
<tr>
<td width="50%">

**Performance**
- Up to 175x faster than [POT](https://pythonot.github.io/) on typical workloads
- Triton fused Sinkhorn -- single-pass logsumexp, zero N*K intermediates
- Mixed precision: float32 Sinkhorn + float64 output
- Smart early stopping via cost plateau detection

</td>
<td width="50%">

**Features**
- Pure GW, Fused GW, and semi-relaxed transport
- Three distance modes: precomputed, Dijkstra, landmark
- Differentiable transport plans (autograd support)
- Low-rank Sinkhorn for N, K > 50k
- Multi-scale coarse-to-fine warm start

</td>
</tr>
</table>

## News

> **v0.4.0** (2026-04-07)  --  Triton fused Sinkhorn (2-5x GPU speedup), mixed precision,
> smart early stopping, Sinkhorn warm-start, Dijkstra caching, and 15 numerical stability fixes.
> See [CHANGELOG.md](CHANGELOG.md).

---

## Installation

```bash
pip install -e .
```

**Requirements:** `numpy`, `scipy`, `scikit-learn`, `torch>=2.0`, `joblib`.
Triton ships with PyTorch and enables GPU kernel fusion automatically. No POT needed.

## Quick Start

```python
from torchgw import sampled_gw

# Basic usage
T = sampled_gw(X_source, X_target)

# Recommended for large-scale (fastest)
T = sampled_gw(X_source, X_target, distance_mode="landmark", mixed_precision=True)
```

<details>
<summary><b>Minimal working example</b> (click to expand)</summary>

```python
import torch
from torchgw import sampled_gw

X = torch.randn(500, 3)   # source: 500 points in 3D
Y = torch.randn(600, 5)   # target: 600 points in 5D (dimensions may differ)

T = sampled_gw(X, Y, epsilon=0.005, M=80, max_iter=200)
# T is a (500, 600) transport plan: T[i,j] = coupling weight between X[i] and Y[j]
print(f"Transport plan: {T.shape}, total mass: {T.sum():.4f}")
```

</details>

## Benchmark

Spiral (2D) to Swiss roll (3D) alignment on NVIDIA L40S:

| Scale | Method | Time | Spearman rho |
|:------|:-------|-----:|:------------:|
| 400 vs 500 | POT `ot.gromov_wasserstein` | 1.6 s | 0.999 |
| 400 vs 500 | **TorchGW** | **0.46 s** | 0.999 |
| 4000 vs 5000 | POT `ot.gromov_wasserstein` | 183 s | 0.999 |
| 4000 vs 5000 | **TorchGW** precomputed | **5.1 s** | 0.998 |
| 4000 vs 5000 | **TorchGW** landmark | **1.0 s** | 0.999 |

> At 4000x5000 with landmark distances, TorchGW is **up to ~175x faster** than POT with equal quality.

<details>
<summary>Benchmark plots</summary>

| 400 vs 500 | 4000 vs 5000 |
|:---:|:---:|
| ![400v500](docs/demo_spiral_to_swissroll_400v500.png) | ![4000v5000](docs/demo_spiral_to_swissroll_4000v5000.png) |

</details>

---

## Distance Modes

Choose based on your data scale:

| Mode | Best for | Per-iteration | Memory | Notes |
|:-----|:--------:|:-------------:|:------:|:------|
| `"precomputed"` | N < 5k | O(NM) lookup | O(N^2) | All-pairs Dijkstra upfront |
| `"dijkstra"` | 5k-50k | O(MN log N) | O(NM) | On-the-fly with caching |
| **`"landmark"`** | **any scale** | O(NMd) GPU | O(Nd) | **Recommended default** |

```python
# Small scale: precompute all distances once
T = sampled_gw(X, Y, distance_mode="precomputed")

# Bring your own distance matrices
T = sampled_gw(dist_source=D_X, dist_target=D_Y, distance_mode="precomputed")

# Large scale (recommended)
T = sampled_gw(X, Y, distance_mode="landmark", n_landmarks=50)
```

---

## Usage Guide

<details>
<summary><b>Best performance settings</b></summary>

```python
T = sampled_gw(
    X, Y,
    distance_mode="landmark",   # avoids expensive all-pairs Dijkstra
    mixed_precision=True,       # float32 Sinkhorn (2x faster on GPU)
    M=80,                       # more samples = better cost estimate
    epsilon=0.005,              # moderate regularization
)
```

</details>

<details>
<summary><b>Fused Gromov-Wasserstein</b></summary>

Blend structural (graph distance) and feature (linear) costs:

```python
C_feat = torch.cdist(features_src, features_tgt)
T = sampled_gw(X, Y, fgw_alpha=0.5, C_linear=C_feat)

# Pure Wasserstein (no graph distances needed)
T = sampled_gw(fgw_alpha=1.0, C_linear=C_feat)
```

</details>

<details>
<summary><b>Semi-relaxed transport</b></summary>

For unbalanced datasets (e.g., cell types present in one sample but not the other):

```python
T = sampled_gw(X, Y, semi_relaxed=True, rho=1.0)
# Source marginal enforced, target marginal relaxed via KL penalty
```

</details>

<details>
<summary><b>Multi-scale warm start</b></summary>

Speeds up convergence by solving a coarse problem first:

```python
T = sampled_gw(X, Y, multiscale=True, n_coarse=200)
```

> Note: GW has symmetric local optima. Works best on data without strong symmetries.

</details>

<details>
<summary><b>Differentiable mode</b></summary>

Use GW cost as a training loss (gradients via envelope theorem):

```python
C_feat = torch.cdist(encoder(X), encoder(Y))
T = sampled_gw(fgw_alpha=1.0, C_linear=C_feat, differentiable=True)
loss = (C_feat.detach() * T).sum()
loss.backward()  # gradients flow to encoder parameters
```

</details>

<details>
<summary><b>Low-rank Sinkhorn (N, K > 50k)</b></summary>

For very large problems where the N*K transport plan does not fit in memory:

```python
from torchgw import sampled_lowrank_gw
T = sampled_lowrank_gw(X, Y, rank=30, distance_mode="landmark", n_landmarks=50)
```

Memory: O((N+K)*rank) instead of O(NK).

</details>

---

## API

### `sampled_gw`

```python
sampled_gw(
    X_source, X_target,         # (N, D) and (K, D') feature matrices
    *,
    distance_mode="dijkstra",   # "precomputed" | "dijkstra" | "landmark"
    fgw_alpha=0.0,              # 0 = pure GW, 1 = Wasserstein, (0,1) = Fused GW
    C_linear=None,              # (N, K) feature cost matrix for FGW
    M=50,                       # anchor pairs per iteration
    epsilon=0.001,              # entropic regularization
    max_iter=500, tol=1e-5,     # convergence control
    mixed_precision=False,      # float32 Sinkhorn for GPU speed
    semi_relaxed=False,         # relax target marginal
    differentiable=False,       # keep autograd graph
    multiscale=False,           # coarse-to-fine warm start
    log=False,                  # return (T, info_dict)
    ...                         # see docs for full parameter list
) -> Tensor                     # (N, K) transport plan
```

### `sampled_lowrank_gw`

Same interface plus `rank`, `lr_max_iter`, `lr_dykstra_max_iter`.
Uses [Scetbon, Cuturi & Peyre (2021)](https://arxiv.org/abs/2103.04737) factorization.

> **When to use:** only when N*K exceeds GPU memory. At smaller scales, `sampled_gw` is faster.

Full API documentation: [chansigit.github.io/torchgw](https://chansigit.github.io/torchgw/)

---

## How It Works

```
                    ┌─────────────────────────────────────────────┐
                    │              GW Main Loop                   │
                    │                                             │
  T_init ──────────►│  1. GPU multinomial sampling (M anchors)    │
                    │  2. Distance computation (Dijkstra/landmark)│
                    │  3. GW cost matrix assembly                 │
                    │  4. Triton fused Sinkhorn projection        │
                    │  5. Momentum update + warm-start            │
                    │  6. Cost plateau convergence check          │
                    │                                             │
                    │  ↺ repeat until converged                   │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                                     T* (N × K)
```

**Acceleration stack:**
- **Triton kernels** -- fused row/column logsumexp, fused T materialization, fused marginal check
- **Warm-start** -- reuse Sinkhorn potentials across iterations
- **Mixed precision** -- float32 log-domain + float64 output
- **Dijkstra cache** -- avoid redundant SSSP across iterations
- **Cost plateau early stopping** -- stop when converged, not at max_iter

---

## Development

```bash
git clone https://github.com/chansigit/torchgw.git
cd torchgw
pip install -e ".[dev]"
pytest tests/ -v          # 72 tests, ~18s
```

## Citation

If you use TorchGW in your research, please cite:

```bibtex
@software{torchgw,
  author = {Sijie Chen},
  title = {TorchGW: Fast Sampled Gromov-Wasserstein Optimal Transport},
  url = {https://github.com/chansigit/torchgw},
  version = {0.4.0},
  year = {2026},
}
```

## License

This project is source-available.

It is free for academic and other non-commercial research and educational use under the terms of the included [`LICENSE`](LICENSE).

Any commercial use — including any use by or on behalf of a for-profit entity, internal commercial research, product development, consulting, paid services, or deployment in commercial settings — requires a separate paid commercial license.

Copyright (c) 2026 The Board of Trustees of the Leland Stanford Junior University.

For commercial licensing inquiries, contact Stanford Office of Technology Licensing: otl@stanford.edu

See [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) for details.
