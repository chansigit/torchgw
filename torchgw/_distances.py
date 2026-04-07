from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

_DIJKSTRA_PARALLEL_THRESHOLD = 64


class DistanceProvider(Protocol):
    """Interface for distance providers used by the TorchGW solver."""

    def get_distances(
        self,
        src_indices: np.ndarray,
        tgt_indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def _batch_dijkstra(graph: csr_matrix, sources: np.ndarray, parallel: bool) -> np.ndarray:
    """Run Dijkstra from multiple sources. Returns (len(sources), N) array."""
    if not parallel or len(sources) < _DIJKSTRA_PARALLEL_THRESHOLD:
        return dijkstra(csgraph=graph, directed=False, indices=sources)
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(dijkstra)(graph, directed=False, indices=int(s)) for s in sources
    )
    return np.vstack(results)


class DijkstraProvider:
    """Compute distances on-the-fly via Dijkstra on kNN graphs."""

    def __init__(self, graph_source: csr_matrix, graph_target: csr_matrix):
        self.graph_source = graph_source
        self.graph_target = graph_target
        self._parallel = max(graph_source.shape[0], graph_target.shape[0]) > 1000

    def get_distances(
        self,
        src_indices: np.ndarray,
        tgt_indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (D_X, D_Y) as float32 tensors on device.

        D_X : (N, len(src_indices))
        D_Y : (K, len(tgt_indices))
        """
        unique_src, src_inv = np.unique(src_indices, return_inverse=True)
        unique_tgt, tgt_inv = np.unique(tgt_indices, return_inverse=True)

        D_src_all = _batch_dijkstra(self.graph_source, unique_src, self._parallel)
        D_tgt_all = _batch_dijkstra(self.graph_target, unique_tgt, self._parallel)

        D_X = torch.from_numpy(D_src_all).float().to(device).T[:, src_inv]
        D_Y = torch.from_numpy(D_tgt_all).float().to(device).T[:, tgt_inv]

        return D_X, D_Y


class PrecomputedProvider:
    """Look up distances from precomputed full pairwise matrices."""

    def __init__(
        self,
        dist_source: torch.Tensor | None = None,
        dist_target: torch.Tensor | None = None,
        graph_source: csr_matrix | None = None,
        graph_target: csr_matrix | None = None,
    ):
        if dist_source is not None and dist_target is not None:
            self.C_X = dist_source.float()
            self.C_Y = dist_target.float()
        elif graph_source is not None and graph_target is not None:
            C_X_np = dijkstra(csgraph=graph_source, directed=False)
            C_Y_np = dijkstra(csgraph=graph_target, directed=False)
            self.C_X = torch.from_numpy(C_X_np).float()
            self.C_Y = torch.from_numpy(C_Y_np).float()
        else:
            raise ValueError(
                "PrecomputedProvider requires either (dist_source, dist_target) "
                "or (graph_source, graph_target)"
            )
        self._cached_device: torch.device | None = None

    def get_distances(
        self,
        src_indices: np.ndarray,
        tgt_indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_device != device:
            self.C_X = self.C_X.to(device)
            self.C_Y = self.C_Y.to(device)
            self._cached_device = device
        D_X = self.C_X[:, src_indices]
        D_Y = self.C_Y[:, tgt_indices]
        return D_X, D_Y


def _landmark_embed(graph: csr_matrix, n_landmarks: int) -> np.ndarray:
    """Compute landmark distance embedding via farthest-point sampling.

    Selects n_landmarks well-spread landmark nodes, then returns the
    (N, n_landmarks) matrix of shortest-path distances from every node
    to each landmark. This serves as a coordinate system where Euclidean
    distance approximates geodesic distance on the graph.

    Parameters
    ----------
    graph : csr_matrix
        Sparse distance graph (must be connected).
    n_landmarks : int
        Number of landmark nodes to select.

    Returns
    -------
    L : ndarray of shape (N, n_landmarks), float32
    """
    N = graph.shape[0]
    n_landmarks = min(n_landmarks, N)

    all_dists = []
    min_dists = np.full(N, np.inf)
    next_idx = 0

    for _ in range(n_landmarks):
        dists = dijkstra(csgraph=graph, directed=False, indices=int(next_idx)).ravel()
        all_dists.append(dists)
        min_dists = np.minimum(min_dists, dists)
        # Handle inf (disconnected nodes) for argmax
        candidates = min_dists.copy()
        candidates[np.isinf(candidates)] = -1.0
        next_idx = int(np.argmax(candidates))

    return np.column_stack(all_dists).astype(np.float32)


class LandmarkProvider:
    """Approximate graph distances via Euclidean distance in landmark-distance space.

    Precomputes distances from all nodes to a set of landmark nodes
    selected by farthest-point sampling. At query time, uses Euclidean
    distance in this landmark-coordinate space as a proxy for geodesic distance.
    """

    def __init__(self, graph_source: csr_matrix, graph_target: csr_matrix, n_landmarks: int = 50):
        self.Z_X = torch.from_numpy(_landmark_embed(graph_source, n_landmarks))
        self.Z_Y = torch.from_numpy(_landmark_embed(graph_target, n_landmarks))

    def get_distances(
        self,
        src_indices: np.ndarray,
        tgt_indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Z_X = self.Z_X.to(device)
        Z_Y = self.Z_Y.to(device)

        D_X = torch.cdist(Z_X, Z_X[src_indices])
        D_Y = torch.cdist(Z_Y, Z_Y[tgt_indices])

        return D_X, D_Y
