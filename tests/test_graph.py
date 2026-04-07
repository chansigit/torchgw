import numpy as np
from scipy.sparse import issparse
from scipy.sparse.csgraph import connected_components

from torchgw._graph import build_knn_graph


def test_build_knn_graph_returns_sparse(two_clusters):
    graph = build_knn_graph(two_clusters, k=10)
    assert issparse(graph)
    assert graph.shape == (200, 200)


def test_build_knn_graph_connected(two_clusters):
    graph = build_knn_graph(two_clusters, k=10)
    n_components, _ = connected_components(graph, directed=False)
    assert n_components == 1


def test_build_knn_graph_small_k_still_connected(two_clusters):
    """Even with k=2 on separated clusters, stitching should connect them."""
    graph = build_knn_graph(two_clusters, k=2)
    n_components, _ = connected_components(graph, directed=False)
    assert n_components == 1


def test_build_knn_graph_symmetric(two_clusters):
    """Graph should be symmetric after .maximum(.T) symmetrization."""
    graph = build_knn_graph(two_clusters, k=10)
    diff = abs(graph - graph.T)
    assert diff.nnz == 0 or diff.max() < 1e-10
