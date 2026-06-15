"""One-step CPU/CUDA smoke test for a Linux Jittor environment."""

import os
from types import SimpleNamespace

import jittor as jt
import numpy as np

from datagen.generate import process_one_voxel
from train._utils import assemble_F_fast, hexahedron, isotropic_elastic_tensor
from train.dataset import batch_to_jittor, sparse_collate
from train.model import GMT
from train.sp_lightning import (
    _assert_finite_gradients,
    _assert_finite_parameters,
    _clip_grad_norm_float64,
    residual_loss,
)


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def main():
    use_cuda = os.environ.get("JITTOR_USE_CUDA", "0") == "1" and jt.has_cuda
    jt.flags.use_cuda = int(use_cuda)
    voxel = np.zeros((4, 4, 4), dtype=bool)
    voxel[:2, :2, :2] = True
    sample = process_one_voxel(voxel, 4)
    batch = batch_to_jittor(sparse_collate([sample]))

    cfg = ns(
        resolution=4,
        model=ns(
            trans_channel=[12, 24, 48],
            trans_block=[1, 1, 1],
            trans_head=[1, 2, 4],
            trans_window_size=[8, 8, 8],
        ),
        GMG=ns(max_cycle=1, smooth_iter=[1, 1, 1]),
    )
    model = GMT(cfg)
    Ke, Fe = hexahedron(4, isotropic_elastic_tensor(1.0, 0.3))
    F = assemble_F_fast(batch["node_index"], Fe, batch["feature"].shape[0])
    u, residual = model(batch, Ke, F)
    loss = residual_loss(residual, batch["coord"], 1)
    optimizer = jt.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.backward(loss)
    _assert_finite_gradients(model, optimizer)
    _clip_grad_norm_float64(model, optimizer, 1.0)
    optimizer.step()
    _assert_finite_parameters(model)

    assert u.shape == (batch["feature"].shape[0], 3, 6)
    assert residual.shape == u.shape
    print(f"smoke-test-ok device={'cuda' if use_cuda else 'cpu'} loss={float(loss.item()):.6f}")


if __name__ == "__main__":
    main()
