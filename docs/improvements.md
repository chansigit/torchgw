# TorchGW — Future Directions

## Potential improvements

### Warm-start quality

`multiscale=True` uses FPS downsampling + coarse solve to warm-start the fine solve.
On data with strong symmetries (e.g., spiral-to-Swiss-roll), the coarse solve may
converge to a different local optimum (e.g., reversed matching), which the fine solve
then inherits. Possible mitigations:

- Run coarse solve with multiple random seeds, pick lowest GW cost
- Detect symmetry-induced reversals and flip automatically
- Use soft assignment (kernel-based) instead of hard nearest-representative

### Low-rank Sinkhorn speed

`sampled_lowrank_gw` is a memory optimization (O((N+K)r) vs O(NK)), not a speed
optimization. At 400x500 it is ~10x slower than standard Sinkhorn due to the
mirror descent + Dykstra overhead. Possible directions:

- Warm-start the low-rank factors (Q, R, g) across SGW iterations instead of
  reinitializing each time
- Use the low-rank factored form directly for sampling and GW cost computation,
  avoiding full T reconstruction
- GPU-optimized Dykstra projection kernels

### torch.compile integration

`_sinkhorn_iterations` is designed to be compilable via `torch.compile(mode="reduce-overhead")`,
which fuses the logsumexp + broadcast operations into fewer GPU kernels. The code includes
a lazy-init compiled path (`_get_compiled_sinkhorn`), but some environments lack the required
`torch._thread_safe_fork` module (observed with PyTorch 2.6 on shared clusters). When compile
is unavailable, the fallback Python loop is used automatically.

Priority: high impact (Sinkhorn is ~73% of loop time). When working, expect 2-4x speedup on
the Sinkhorn portion. Fix: ensure a complete PyTorch installation with Inductor support.

### Triton fused Sinkhorn kernel

A custom Triton kernel for the Sinkhorn row/column updates can achieve the same fusion as
`torch.compile` without depending on the Inductor compiler. The kernel would perform single-pass
online logsumexp (no intermediate N×K matrix), reducing both memory and kernel launch overhead.
This is the recommended path if `torch.compile` remains unavailable.

### GPU-native shortest paths (cuGraph)

`DijkstraProvider` runs scipy Dijkstra on CPU. For `distance_mode="dijkstra"`, this is the
main per-iteration cost. `nvidia/cugraph` provides GPU-native SSSP but is hard to install
(requires RAPIDS conda environment). Implementation plan:

- Add optional `backend="scipy"` / `"cugraph"` parameter to distance providers
- Lazy-import cugraph at runtime; fall back to scipy if unavailable
- Do NOT add cugraph as a hard dependency

### Scalability beyond 100k

At N, K > 100k the full N x K cost matrix Lambda becomes the memory bottleneck
(not just the transport plan). Combining low-rank Sinkhorn with landmark distances
avoids storing T, but Lambda is still assembled each iteration. Possible directions:

- Stochastic cost matrix: only compute a random subset of Lambda entries each iteration
- Implicit cost via low-rank distance factorization (C = M1 @ M2^T for squared Euclidean)
- Block-coordinate descent on the transport plan
