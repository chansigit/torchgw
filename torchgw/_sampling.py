import numpy as np
import torch


def sample_pairs_from_plan(
    T: np.ndarray, M: int, rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample M (row, col) pairs from transport plan T, weighted by mass.

    Uses the Gumbel-max trick for vectorized categorical sampling,
    avoiding a Python for-loop over M pairs.

    Parameters
    ----------
    T : ndarray of shape (N, K), non-negative
    M : number of pairs to sample
    rng : numpy Generator, optional
        Random number generator for reproducibility. Uses a new default
        generator if None.

    Returns
    -------
    rows : ndarray of shape (M,)
    cols : ndarray of shape (M,)
    """
    if rng is None:
        rng = np.random.default_rng()

    N, K = T.shape
    p_rows = T.sum(axis=1)
    total = p_rows.sum()
    if total < 1e-9:
        rows = rng.integers(0, N, size=M)
        cols = rng.integers(0, K, size=M)
        return rows, cols

    p_rows = p_rows / total
    sampled_rows = rng.choice(N, size=M, p=p_rows)

    # Gumbel-max trick: vectorized categorical sampling over columns
    row_slices = T[sampled_rows]  # (M, K)
    row_sums = row_slices.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-30)
    log_probs = np.log(row_slices / row_sums + 1e-30)
    gumbel_noise = -np.log(-np.log(rng.uniform(size=(M, K)) + 1e-30) + 1e-30)
    sampled_cols = np.argmax(log_probs + gumbel_noise, axis=1)

    return sampled_rows, sampled_cols


def sample_pairs_gpu(
    T: torch.Tensor, M: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample M (row, col) pairs on GPU using torch.multinomial.

    Avoids transferring the full (N, K) transport plan to CPU.
    Only transfers 2*M integers back.

    Parameters
    ----------
    T : Tensor of shape (N, K), non-negative, on any device
    M : number of pairs to sample

    Returns
    -------
    rows : ndarray of shape (M,)
    cols : ndarray of shape (M,)
    """
    N, K = T.shape

    # Row sampling
    p_rows = T.sum(dim=1)
    total = p_rows.sum()
    if total < 1e-9:
        rng = np.random.default_rng()
        return rng.integers(0, N, size=M), rng.integers(0, K, size=M)

    p_rows = p_rows / total
    rows = torch.multinomial(p_rows, M, replacement=True)

    # Column sampling per selected row
    row_probs = T[rows]  # (M, K)
    row_sums = row_probs.sum(dim=1, keepdim=True).clamp(min=1e-30)
    row_probs = row_probs / row_sums
    cols = torch.multinomial(row_probs, 1).squeeze(1)

    return rows.cpu().numpy(), cols.cpu().numpy()
