"""Jittor FEM utilities used by the GMT training loop."""

import jittor as jt
import numpy as np


def scatter_add(src: jt.Var, index: jt.Var, dim_size: int) -> jt.Var:
    out = jt.zeros((dim_size,) + tuple(src.shape[1:]), dtype=src.dtype)
    index_shape = (index.shape[0],) + (1,) * (len(src.shape) - 1)
    expanded_index = index.int32().reshape(index_shape).broadcast(src.shape)
    return out.scatter(0, expanded_index, src, reduce="add")


def scatter_mean(src: jt.Var, index: jt.Var, dim_size: int) -> jt.Var:
    out = scatter_add(src, index, dim_size)
    count = scatter_add(
        jt.ones((src.shape[0], 1), dtype=src.dtype), index, dim_size
    )
    return out / jt.maximum(count, jt.array(1.0, dtype=src.dtype))


def isotropic_elastic_tensor(E: float, poisson: float) -> jt.Var:
    lam = poisson / (1.0 + poisson) / (1.0 - 2.0 * poisson) * E
    mu = E / (2.0 * (1.0 + poisson))
    return jt.array(
        [
            [lam + 2 * mu, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mu, lam, 0, 0, 0],
            [lam, lam, lam + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ]
    ).float32()


def hexahedron(resolution: int, C: jt.Var):
    """Build the trilinear hexahedron stiffness and macro-strain force matrices."""
    # This matrix is constant and small. NumPy avoids creating a large Jittor graph
    # during startup while preserving the exact FEM integration.
    C_np = C.numpy().astype(np.float64)
    half = 1.0 / resolution / 2.0
    pp = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
    ww = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    dxdydz = np.array(
        [
            [-half, half, half, -half, -half, half, half, -half],
            [-half, -half, half, half, -half, -half, half, half],
            [-half, -half, -half, -half, half, half, half, half],
        ]
    ).T
    Ke = np.zeros((24, 24), dtype=np.float64)
    Fe = np.zeros((24, 6), dtype=np.float64)

    for i, x in enumerate(pp):
        for j, y in enumerate(pp):
            for k, z in enumerate(pp):
                q = np.array(
                    [
                        [
                            -((y - 1) * (z - 1)) / 8,
                            ((y - 1) * (z - 1)) / 8,
                            -((y + 1) * (z - 1)) / 8,
                            ((y + 1) * (z - 1)) / 8,
                            ((y - 1) * (z + 1)) / 8,
                            -((y - 1) * (z + 1)) / 8,
                            ((y + 1) * (z + 1)) / 8,
                            -((y + 1) * (z + 1)) / 8,
                        ],
                        [
                            -((x - 1) * (z - 1)) / 8,
                            ((x + 1) * (z - 1)) / 8,
                            -((x + 1) * (z - 1)) / 8,
                            ((x - 1) * (z - 1)) / 8,
                            ((x - 1) * (z + 1)) / 8,
                            -((x + 1) * (z + 1)) / 8,
                            ((x + 1) * (z + 1)) / 8,
                            -((x - 1) * (z + 1)) / 8,
                        ],
                        [
                            -((x - 1) * (y - 1)) / 8,
                            ((x + 1) * (y - 1)) / 8,
                            -((x + 1) * (y + 1)) / 8,
                            ((x - 1) * (y + 1)) / 8,
                            ((x - 1) * (y - 1)) / 8,
                            -((x + 1) * (y - 1)) / 8,
                            ((x + 1) * (y + 1)) / 8,
                            -((x - 1) * (y + 1)) / 8,
                        ],
                    ]
                )
                J = q @ dxdydz
                qxyz = np.linalg.inv(J) @ q
                B = np.zeros((6, 24), dtype=np.float64)
                for node in range(8):
                    dx, dy, dz = qxyz[:, node]
                    B[:, node * 3 : (node + 1) * 3] = [
                        [dx, 0, 0],
                        [0, dy, 0],
                        [0, 0, dz],
                        [dy, dx, 0],
                        [0, dz, dy],
                        [dz, 0, dx],
                    ]
                weight = np.linalg.det(J) * ww[i] * ww[j] * ww[k]
                Ke += weight * B.T @ C_np @ B
                Fe += weight * B.T @ C_np

    return jt.array(Ke.astype(np.float32)), jt.array(Fe.astype(np.float32))


def build_dof_indices(node_index: jt.Var) -> jt.Var:
    base = node_index.int32().unsqueeze(-1) * 3
    offsets = jt.array([0, 1, 2]).int32().reshape(1, 1, 3)
    return (base + offsets).reshape(node_index.shape[0], 24)


def assemble_F_fast(node_index: jt.Var, Fe: jt.Var, n_nodes: int) -> jt.Var:
    values = Fe.reshape(8, 3, 6).unsqueeze(0).broadcast(
        (node_index.shape[0], 8, 3, 6)
    )
    return scatter_add(values.reshape(-1, 3, 6), node_index.reshape(-1), n_nodes)


def ebe_matvec(Ke: jt.Var, u: jt.Var, dof_indices: jt.Var) -> jt.Var:
    """Matrix-free global K*u for one or more right hand sides."""
    n_nodes = u.shape[0]
    n_cols = u.shape[2] if len(u.shape) == 3 else 1
    u_work = u.reshape(n_nodes * 3, n_cols)
    u_elem = u_work[dof_indices.reshape(-1)].reshape(-1, 24, n_cols)
    # Jittor's CUDA batched matmul does not broadcast Ke's leading dimension.
    # Flatten elements and right hand sides so one 2-D matmul applies Ke to all.
    y_elem = jt.matmul(
        u_elem.transpose(0, 2, 1).reshape(-1, 24), Ke.transpose(1, 0)
    ).reshape(-1, n_cols, 24).transpose(0, 2, 1)
    y = scatter_add(y_elem.reshape(-1, n_cols), dof_indices.reshape(-1), n_nodes * 3)
    return y.reshape(n_nodes, 3, n_cols)


def ebe_diagonal(Ke: jt.Var, dof_indices: jt.Var, n_nodes: int) -> jt.Var:
    diag = jt.diag(Ke)
    values = diag.unsqueeze(0).broadcast((dof_indices.shape[0], 24))
    out = scatter_add(values.reshape(-1, 1), dof_indices.reshape(-1), n_nodes * 3)
    return out.reshape(n_nodes, 3, 1)
