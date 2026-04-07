import inspect

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds, cg, LinearOperator

# SciPy >= 1.12 renamed cg(tol=) to cg(rtol=); detect which to use.
_CG_TOL_KEY = "rtol" if "rtol" in inspect.signature(cg).parameters else "tol"


def joint_embedding(
    anchor_name: str,
    data_by_name: dict[str, np.ndarray],
    graphs_by_name: dict[str, csr_matrix],
    transport_plans: dict[tuple[str, str], np.ndarray],
    lambda_reg: float = 1.0,
    out_dim: int = 30,
) -> dict[str, np.ndarray]:
    """Compute joint manifold embedding using transport plans.

    Parameters
    ----------
    anchor_name : str
        Name of the anchor/reference dataset.
    data_by_name : dict mapping name -> (n_samples, n_features) array
    graphs_by_name : dict mapping name -> kNN graph (csr_matrix)
    transport_plans : dict mapping (query_name, anchor_name) -> T array
    lambda_reg : float
        Regularization weight for Laplacian.
    out_dim : int
        Dimensionality of output embedding.

    Returns
    -------
    embeddings : dict mapping name -> (n_samples, out_dim) array
    """
    query_names = [n for n in data_by_name if n != anchor_name]

    # Build graph Laplacians
    laplacians = {}
    for name, graph in graphs_by_name.items():
        W = graph.copy()
        W.data = 1.0 / (W.data + 1e-10)
        W = W.maximum(W.T)
        D = diags(W.sum(axis=1).A1)
        laplacians[name] = (D - W).asformat("csr")

    T_list = [transport_plans[(name, anchor_name)] for name in query_names]
    Sigma_x_list = [diags(T.sum(axis=1)) for T in T_list]
    Sigma_y_list = [diags(T.sum(axis=0)) for T in T_list]

    S_xx_blocks = [
        laplacians[name] + lambda_reg * Sigma_x_list[i]
        for i, name in enumerate(query_names)
    ]
    S_yy = laplacians[anchor_name] + lambda_reg * sum(Sigma_y_list)

    N_q = sum(b.shape[0] for b in S_xx_blocks)
    N_a = S_yy.shape[0]

    diag_yy = S_yy.diagonal().copy()
    diag_yy[diag_yy == 0] = 1.0
    precond_y = diags(1.0 / diag_yy)

    def _cg_worker(block, v_chunk):
        d = block.diagonal().copy()
        d[d == 0] = 1.0
        precond = diags(1.0 / d)
        x, _ = cg(block, v_chunk.astype(np.float32), M=precond, **{_CG_TOL_KEY: 1e-4})
        return x.ravel()

    def H_x_matvec(v):
        v = v.ravel()
        chunks = []
        pos = 0
        for block in S_xx_blocks:
            n = block.shape[0]
            chunks.append(v[pos:pos + n])
            pos += n
        results = Parallel(n_jobs=-1)(
            delayed(_cg_worker)(S_xx_blocks[i], chunks[i])
            for i in range(len(S_xx_blocks))
        )
        return np.concatenate(results)

    def H_y_matvec(v):
        x, _ = cg(S_yy, v.ravel().astype(np.float32), M=precond_y, **{_CG_TOL_KEY: 1e-4})
        return x.ravel()

    def H_matvec(v):
        y_mult = H_y_matvec(v)
        xy = np.zeros(N_q, dtype=np.float32)
        pos = 0
        for i, T in enumerate(T_list):
            n = T.shape[0]
            xy[pos:pos + n] = T @ y_mult
            pos += n
        return H_x_matvec(xy)

    def H_rmatvec(v):
        x_mult = H_x_matvec(v)
        xy_T = np.zeros(N_a, dtype=np.float32)
        pos = 0
        for i, T in enumerate(T_list):
            n = T.shape[0]
            xy_T += T.T @ x_mult[pos:pos + n]
            pos += n
        return H_y_matvec(xy_T)

    H_op = LinearOperator((N_q, N_a), matvec=H_matvec, rmatvec=H_rmatvec)

    k_svds = min(out_dim + 2, N_q - 1, N_a - 1)
    actual_dim = min(out_dim, k_svds)
    U, _, Vt = svds(H_op, k=k_svds, tol=1e-4)
    U = U[:, ::-1][:, :actual_dim]
    V = Vt.T[:, ::-1][:, :actual_dim]

    embeddings = {anchor_name: V.astype(np.float32)}
    pos = 0
    for name in query_names:
        n = data_by_name[name].shape[0]
        embeddings[name] = U[pos:pos + n].astype(np.float32)
        pos += n

    return embeddings
