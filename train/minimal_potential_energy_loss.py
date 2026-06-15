import jittor as jt

from train._utils import build_dof_indices, ebe_matvec


def minimal_potential_energy(Ke, F, u, node_index):
    Ku = ebe_matvec(Ke, u.unsqueeze(-1), build_dof_indices(node_index)).squeeze(-1)
    return (u * (0.5 * Ku - F)).sum() / u.shape[0]


def displacement_regularization(u, coord, batch_size):
    loss = jt.array(0.0)
    for batch_id in range(batch_size):
        mean = u[coord[:, 0] == batch_id].mean(dim=0)
        loss = loss + (mean * mean).mean()
    return loss / batch_size
