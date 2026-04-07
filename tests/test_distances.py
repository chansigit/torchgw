import numpy as np
import pytest
import torch
from torchgw._graph import build_knn_graph
from torchgw._distances import DijkstraProvider, PrecomputedProvider, LandmarkProvider


def test_dijkstra_provider_shapes():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    Y = rng.normal(size=(60, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=10)
    g_y = build_knn_graph(Y, k=10)

    provider = DijkstraProvider(g_x, g_y)
    src_idx = np.array([0, 5, 10])
    tgt_idx = np.array([1, 3, 7, 9])
    device = torch.device("cpu")

    D_X, D_Y = provider.get_distances(src_idx, tgt_idx, device)

    assert D_X.shape == (50, 3)
    assert D_Y.shape == (60, 4)
    assert D_X.dtype == torch.float32
    assert D_Y.dtype == torch.float32
    assert D_X.device.type == "cpu"


def test_dijkstra_provider_nonnegative():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=10)
    g_y = build_knn_graph(X.copy(), k=10)

    provider = DijkstraProvider(g_x, g_y)
    D_X, D_Y = provider.get_distances(np.array([0, 1]), np.array([0, 1]), torch.device("cpu"))

    assert torch.all(D_X >= 0)
    assert torch.all(D_Y >= 0)


def test_precomputed_provider_from_matrices():
    """User passes dist_source and dist_target directly."""
    rng = np.random.default_rng(42)
    C_X = rng.random((40, 40)).astype(np.float32)
    C_Y = rng.random((50, 50)).astype(np.float32)

    provider = PrecomputedProvider(
        dist_source=torch.from_numpy(C_X),
        dist_target=torch.from_numpy(C_Y),
    )
    src_idx = np.array([0, 5, 10])
    tgt_idx = np.array([1, 3])
    device = torch.device("cpu")

    D_X, D_Y = provider.get_distances(src_idx, tgt_idx, device)

    assert D_X.shape == (40, 3)
    assert D_Y.shape == (50, 2)
    # Should be exact column indexing
    torch.testing.assert_close(D_X[:, 0], torch.from_numpy(C_X[:, 0]))
    torch.testing.assert_close(D_Y[:, 1], torch.from_numpy(C_Y[:, 3]))


def test_precomputed_provider_from_graphs():
    """User doesn't pass dist matrices — provider computes all-pairs Dijkstra."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    Y = rng.normal(size=(35, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=10)
    g_y = build_knn_graph(Y, k=10)

    provider = PrecomputedProvider(graph_source=g_x, graph_target=g_y)
    D_X, D_Y = provider.get_distances(np.array([0, 1]), np.array([0, 1]), torch.device("cpu"))

    assert D_X.shape == (30, 2)
    assert D_Y.shape == (35, 2)
    assert torch.all(torch.isfinite(D_X))


def test_landmark_provider_shapes():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    Y = rng.normal(size=(60, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=10)
    g_y = build_knn_graph(Y, k=10)

    provider = LandmarkProvider(g_x, g_y, n_landmarks=10)
    src_idx = np.array([0, 5, 10])
    tgt_idx = np.array([1, 3, 7, 9])
    device = torch.device("cpu")

    D_X, D_Y = provider.get_distances(src_idx, tgt_idx, device)

    assert D_X.shape == (50, 3)
    assert D_Y.shape == (60, 4)
    assert D_X.dtype == torch.float32
    assert torch.all(D_X >= 0)
    assert torch.all(D_Y >= 0)


def test_dijkstra_provider_cache_hit():
    """Calling get_distances twice with overlapping indices should use cache."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=10)
    g_y = build_knn_graph(X.copy(), k=10)

    provider = DijkstraProvider(g_x, g_y)
    device = torch.device("cpu")

    # First call populates cache
    idx1 = np.array([0, 5, 10])
    D_X1, D_Y1 = provider.get_distances(idx1, idx1, device)
    assert len(provider._cache_src) == 3

    # Second call with overlapping indices should hit cache
    idx2 = np.array([5, 10, 20])  # 5 and 10 are cached
    D_X2, D_Y2 = provider.get_distances(idx2, idx2, device)
    assert len(provider._cache_src) == 4  # only node 20 is new

    # Cached values should be consistent
    torch.testing.assert_close(D_X1[:, 1], D_X2[:, 0])  # both are node 5


def test_landmark_provider_self_distance_zero():
    """Distance from a point to itself should be zero."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    g_x = build_knn_graph(X, k=10)

    provider = LandmarkProvider(g_x, g_x, n_landmarks=10)
    D_X, _ = provider.get_distances(np.array([5]), np.array([0]), torch.device("cpu"))

    assert D_X[5, 0].item() == pytest.approx(0.0, abs=1e-3)
