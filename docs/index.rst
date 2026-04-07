TorchGW �� Fast Sampled Gromov-Wasserstein Optimal Transport
============================================================

.. image:: logo.svg
   :width: 480
   :align: center
   :alt: TorchGW — Fast Gromov-Wasserstein optimal transport solver in PyTorch

|

TorchGW is a scalable solver for `Gromov-Wasserstein <https://arxiv.org/abs/1805.09114>`_
optimal transport, implemented in **pure PyTorch** with GPU-accelerated
`Triton <https://github.com/triton-lang/triton>`_ fused Sinkhorn kernels.

It aligns two point clouds by matching their internal distance structures — even when
the point clouds live in **different dimensions** — making it ideal for manifold alignment,
single-cell multi-omics integration, and cross-domain graph matching.

**Key features:**

- **3–175x faster than POT** on typical workloads (spiral 4000×5000: 1s vs 183s)
- **Triton fused Sinkhorn** — single-pass online logsumexp, no intermediate N×K matrices
- **Mixed precision** — float32 Sinkhorn + float64 output, zero quality loss
- **Smart early stopping** — cost plateau detection, not just transport plan norm
- **Differentiable** — use GW cost as a training loss with autograd support
- **No POT dependency** at runtime — pure PyTorch + scipy + scikit-learn

What's New in v0.4.0
---------------------

- Triton fused Sinkhorn kernels (2–5x GPU speedup)
- Mixed precision support (``mixed_precision=True``)
- Cost plateau early stopping
- Sinkhorn warm-start across GW iterations
- 15 numerical stability and correctness fixes
- See :doc:`changelog` for details

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api
   algorithm
   benchmark
   changelog
   optimization-log


Installation
------------

.. code-block:: bash

   pip install -e .

Dependencies: ``numpy``, ``scipy``, ``scikit-learn``, ``torch``, ``joblib``.
Triton (ships with PyTorch 2.0+) enables GPU kernel fusion automatically.

Quick Example
-------------

.. code-block:: python

   from torchgw import sampled_gw

   T = sampled_gw(X_source, X_target, distance_mode="landmark", mixed_precision=True)
   # T[i,j] = optimal coupling weight between source point i and target point j

License
-------

MIT
