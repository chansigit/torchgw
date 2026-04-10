TorchGW -- Fast Sampled Gromov-Wasserstein Optimal Transport
============================================================

.. image:: logo.svg
   :width: 480
   :align: center
   :alt: TorchGW — Fast Gromov-Wasserstein optimal transport solver in PyTorch

|

.. image:: https://img.shields.io/badge/github-chansigit%2Ftorchgw-black?logo=github
   :target: https://github.com/chansigit/torchgw
   :alt: GitHub

.. image:: https://img.shields.io/badge/version-0.4.1-green
   :target: https://github.com/chansigit/torchgw/blob/main/CHANGELOG.md
   :alt: Version

.. image:: https://img.shields.io/badge/license-Non--Commercial-orange
   :target: https://github.com/chansigit/torchgw/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white
   :alt: Python

.. image:: https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white
   :target: https://pytorch.org/
   :alt: PyTorch

|

.. note::

   **Source code:** `github.com/chansigit/torchgw <https://github.com/chansigit/torchgw>`_
   — clone, star, or open issues on GitHub.

TorchGW is a scalable solver for `Gromov-Wasserstein <https://arxiv.org/abs/1805.09114>`_
optimal transport, implemented in **pure PyTorch** with GPU-accelerated
`Triton <https://github.com/triton-lang/triton>`_ fused Sinkhorn kernels.

It aligns two point clouds by matching their internal distance structures — even when
the point clouds live in **different dimensions** — making it ideal for manifold alignment,
single-cell multi-omics integration, and cross-domain graph matching.

**Key features:**

- **Up to 175x faster than POT** on typical workloads (spiral 4000×5000: 1s vs 183s)
- **Triton fused Sinkhorn** — single-pass online logsumexp, no intermediate N×K matrices
- **Mixed precision** — float32 Sinkhorn + float64 output, zero quality loss
- **Smart early stopping** — cost plateau detection, not just transport plan norm
- **Differentiable** — exact gradients via implicit differentiation at the Sinkhorn fixed point
- **No POT dependency** at runtime — pure PyTorch + scipy + scikit-learn

What's New in v0.4.1
---------------------

- **Exact differentiable gradients** via implicit differentiation at the Sinkhorn
  fixed point — fixes a correctness bug where the old backward produced gradients
  with up to 30x error
- New ``grad_mode`` parameter: ``"implicit"`` (default, exact) or ``"unrolled"``
- Full theory derivation in :doc:`algorithm`
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

   pip install torchgw

Or for development:

.. code-block:: bash

   git clone https://github.com/chansigit/torchgw.git
   cd torchgw && pip install -e ".[dev]"

Dependencies: ``numpy``, ``scipy``, ``scikit-learn``, ``torch>=2.0``, ``joblib``.
Triton (ships with PyTorch 2.0+) enables GPU kernel fusion automatically.

Quick Example
-------------

.. code-block:: python

   from torchgw import sampled_gw

   T = sampled_gw(X_source, X_target, distance_mode="landmark", mixed_precision=True)
   # T[i,j] = optimal coupling weight between source point i and target point j

Source Code & Links
-------------------

- **GitHub repository:** `chansigit/torchgw <https://github.com/chansigit/torchgw>`_
- **Issue tracker:** `GitHub Issues <https://github.com/chansigit/torchgw/issues>`_
- **Changelog:** `CHANGELOG.md <https://github.com/chansigit/torchgw/blob/main/CHANGELOG.md>`_
- **PyPI:** *coming soon*

.. code-block:: bash

   # Clone and install from source
   git clone https://github.com/chansigit/torchgw.git
   cd torchgw
   pip install -e .

Citation
--------

If you use TorchGW in your research, please cite:

.. code-block:: bibtex

   @software{torchgw,
     author = {Sijie Chen},
     title = {TorchGW: Fast Sampled Gromov-Wasserstein Optimal Transport},
     url = {https://github.com/chansigit/torchgw},
     version = {0.4.1},
     year = {2026},
   }

License
-------

Free for academic and non-commercial use. Commercial use requires a separate license.
See `LICENSE <https://github.com/chansigit/torchgw/blob/main/LICENSE>`_ and
`COMMERCIAL_LICENSE.md <https://github.com/chansigit/torchgw/blob/main/COMMERCIAL_LICENSE.md>`_
for details.

Copyright (c) 2026 The Board of Trustees of the Leland Stanford Junior University.
For commercial licensing inquiries, contact Stanford OTL: otl@stanford.edu
