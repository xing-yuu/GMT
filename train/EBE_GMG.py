"""Matrix-free geometric multigrid components implemented with Jittor."""

from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn

from train._utils import build_dof_indices, ebe_diagonal, ebe_matvec, scatter_mean


def _parent_index(fine_coord: jt.Var, coarse_coord: jt.Var) -> jt.Var:
    fine = fine_coord.numpy().astype(np.int32, copy=False)
    coarse = coarse_coord.numpy().astype(np.int32, copy=False)
    lookup = {tuple(row): index for index, row in enumerate(coarse)}
    parent = fine.copy()
    parent[:, 1:4] //= 2
    indices = np.array([lookup[tuple(row)] for row in parent], dtype=np.int32)
    return jt.array(indices).int32()


def remove_batch_mean(u: jt.Var, coord: jt.Var) -> jt.Var:
    batch = coord[:, 0].int32()
    batch_count = int(batch.numpy().max()) + 1 if batch.shape[0] else 1
    return u - scatter_mean(u, batch, batch_count)[batch]


class EBEOperator(nn.Module):
    """Differentiable element-by-element linear elasticity operator."""

    def execute(self, Ke: jt.Var, u: jt.Var, node_index: jt.Var) -> jt.Var:
        return ebe_matvec(Ke, u.unsqueeze(-1), build_dof_indices(node_index)).squeeze(-1)

    def smooth_jacobi(
        self,
        Ke: jt.Var,
        u: jt.Var,
        f: jt.Var,
        node_index: jt.Var,
        coord: jt.Var,
        iterations: int,
        omega: float = 0.67,
    ) -> jt.Var:
        dof_indices = build_dof_indices(node_index)
        diag = ebe_diagonal(Ke, dof_indices, u.shape[0]).squeeze(-1) + 1e-12
        for _ in range(iterations):
            residual = f - ebe_matvec(Ke, u.unsqueeze(-1), dof_indices).squeeze(-1)
            u = remove_batch_mean(u + omega * residual / diag, coord)
        return u


class GMGSolver(nn.Module):
    """Learned multilevel prolongation followed by an EBE Jacobi correction."""

    def __init__(self):
        self.operator = EBEOperator()

    def execute(
        self,
        fine_ke: jt.Var,
        global_f: jt.Var,
        node_coords_L0: jt.Var,
        fine_topo_indices: jt.Var,
        max_iter: int,
        smooth_iter: List[int],
        init_list: List[Dict],
    ):
        u = init_list[-1]["u"]
        for level in reversed(range(len(init_list) - 1)):
            parent = _parent_index(init_list[level]["coord"], init_list[level + 1]["coord"])
            u = init_list[level]["u"] + u[parent]

        iterations = max(1, int(max_iter)) * max(
            1, sum(int(value) for value in smooth_iter)
        )
        u = self.operator.smooth_jacobi(
            fine_ke,
            u,
            global_f,
            fine_topo_indices,
            node_coords_L0,
            iterations,
        )
        residual = global_f - self.operator(fine_ke, u, fine_topo_indices)
        return u, residual
