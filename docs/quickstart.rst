Quick Start
===========

Installation
------------

.. code-block:: bash

   git clone https://github.com/chansigit/torchgw.git
   cd torchgw
   pip install -e .

Or from PyPI:

.. code-block:: bash

   pip install torchgw

Requirements: ``numpy``, ``scipy``, ``scikit-learn``, ``torch>=2.0``, ``joblib``.
Source code: `github.com/chansigit/torchgw <https://github.com/chansigit/torchgw>`_.

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   from torchgw import sampled_gw, build_knn_graph

   # Two point clouds (dimensions may differ)
   X = np.random.randn(500, 3).astype(np.float32)
   Y = np.random.randn(600, 5).astype(np.float32)

   # Compute transport plan
   T = sampled_gw(X, Y, epsilon=0.005, M=80, max_iter=200)

   # T[i,j] is the coupling weight between X[i] and Y[j]
   print(T.shape)  # (500, 600)

With Precomputed Graphs
-----------------------

Building the kNN graph once and reusing it avoids redundant computation:

.. code-block:: python

   g_src = build_knn_graph(X, k=10)
   g_tgt = build_knn_graph(Y, k=10)

   T = sampled_gw(X, Y, graph_source=g_src, graph_target=g_tgt,
                  epsilon=0.005, M=80, max_iter=200)

Semi-Relaxed Mode
-----------------

When source and target have different compositions (e.g., a cell type
present in source but absent in target), balanced GW forces mass onto
wrong matches. Semi-relaxed GW fixes the source marginal but lets the
target marginal adapt:

.. code-block:: python

   # Balanced (default): T @ 1 = p,  T.T @ 1 = q  (both enforced)
   T = sampled_gw(X, Y, epsilon=0.005)

   # Semi-relaxed: T @ 1 = p (enforced),  T.T @ 1 ≈ q (soft KL penalty)
   T = sampled_gw(X, Y, epsilon=0.005, semi_relaxed=True, rho=1.0)

   # rho controls how strictly q is enforced:
   #   rho → ∞  : recovers balanced GW
   #   rho → 0  : target marginal is completely free

Convergence Logging
-------------------

.. code-block:: python

   T, info = sampled_gw(X, Y, epsilon=0.005, max_iter=200, log=True)
   print(info["n_iter"])    # actual iterations run
   print(info["err_list"])  # per-iteration convergence errors

Differentiable Mode
-------------------

For end-to-end training, keep the computation graph:

.. code-block:: python

   import torch
   from torchgw import sampled_gw

   C_feat = torch.cdist(encoder(X), encoder(Y))

   # Default: exact gradients via implicit differentiation
   T = sampled_gw(fgw_alpha=1.0, C_linear=C_feat, differentiable=True)
   loss = (C_feat.detach() * T).sum()
   loss.backward()  # exact gradients flow to encoder parameters

   # Alternative: unrolled autograd (higher memory, useful as fallback)
   T = sampled_gw(fgw_alpha=1.0, C_linear=C_feat,
                  differentiable=True, grad_mode="unrolled")

The default ``grad_mode="implicit"`` solves the adjoint system at the
Sinkhorn fixed point for exact gradients with O(NK) memory. See
:doc:`algorithm` for the mathematical derivation.

.. note::

   ``differentiable=True`` requires ``fgw_alpha > 0`` with a differentiable
   ``C_linear``.  Pure GW (``fgw_alpha=0``) uses precomputed graph distances
   that are not part of the computation graph; a warning is emitted in this case.

Joint Embedding
---------------

After computing a transport plan, embed both datasets into a shared space:

.. code-block:: python

   from torchgw import joint_embedding

   emb = joint_embedding(
       anchor_name="tgt",
       data_by_name={"src": X, "tgt": Y},
       graphs_by_name={"src": g_src, "tgt": g_tgt},
       transport_plans={("src", "tgt"): T},
       out_dim=10,
   )
   print(emb["src"].shape)  # (500, 10)
   print(emb["tgt"].shape)  # (600, 10)
