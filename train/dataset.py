import os
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    import jittor as jt


class SparseDataset:
    """Dataset for the variable-size sparse voxel samples used by GMT."""

    def __init__(self, base_paths: Sequence[str]):
        self.data_name_set: List[str] = []
        for base_path in base_paths:
            if not os.path.isdir(base_path):
                continue
            names = sorted(os.listdir(base_path))
            for name in tqdm(names, desc=f"Loading {base_path}", leave=False):
                if name.lower().endswith(".npz"):
                    self.data_name_set.append(os.path.join(base_path, name))

    def __len__(self) -> int:
        return len(self.data_name_set)

    def __getitem__(self, index: int):
        with np.load(self.data_name_set[index], allow_pickle=False) as data:
            return (
                data["coords"].astype(np.int32, copy=False),
                data["node_type"].astype(np.float32, copy=False),
                data["node_index"].astype(np.int32, copy=False),
                data["voxel"].astype(bool, copy=False),
            )


def sparse_collate(samples) -> Dict[str, np.ndarray]:
    all_coords = []
    all_feats = []
    all_node_index = []
    all_elem_coords = []
    voxel_batch_id = []
    node_base = 0

    for batch_id, (coords, feats, node_index, voxel) in enumerate(samples):
        n_nodes = coords.shape[0]
        batch_col = np.full((n_nodes, 1), batch_id, dtype=np.int32)
        all_coords.append(np.concatenate([batch_col, coords], axis=1))
        all_feats.append(feats)
        all_node_index.append(node_index + node_base)

        elem_xyz = np.argwhere(voxel).astype(np.int32, copy=False)
        elem_batch = np.full((elem_xyz.shape[0], 1), batch_id, dtype=np.int32)
        all_elem_coords.append(np.concatenate([elem_batch, elem_xyz], axis=1))
        voxel_batch_id.append(np.full((node_index.shape[0],), batch_id, dtype=np.int32))
        node_base += n_nodes

    return {
        "coord": np.concatenate(all_coords, axis=0),
        "feature": np.concatenate(all_feats, axis=0),
        "node_index": np.concatenate(all_node_index, axis=0),
        "elem_coords": np.concatenate(all_elem_coords, axis=0),
        "voxel_batch_id": np.concatenate(voxel_batch_id, axis=0),
    }


def batch_to_jittor(batch: Dict[str, np.ndarray]) -> Dict[str, "jt.Var"]:
    import jittor as jt

    return {
        "coord": jt.array(batch["coord"]).int32(),
        "feature": jt.array(batch["feature"]).float32(),
        "node_index": jt.array(batch["node_index"]).int32(),
        "elem_coords": jt.array(batch["elem_coords"]).int32(),
        "voxel_batch_id": jt.array(batch["voxel_batch_id"]).int32(),
    }


def distributed_indices(
    dataset_size: int,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    rank: int,
    world_size: int,
    seed: int,
) -> np.ndarray:
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.RandomState(seed).shuffle(indices)

    if world_size > 1:
        if drop_last:
            global_batch_size = batch_size * world_size
            usable = len(indices) // global_batch_size * global_batch_size
            indices = indices[:usable]
        indices = indices[rank::world_size]
    return indices


def iter_sparse_batches(
    dataset: SparseDataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = True,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
) -> Iterable[Dict[str, "jt.Var"]]:
    indices = distributed_indices(
        len(dataset), batch_size, shuffle, drop_last, rank, world_size, seed
    )

    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        if len(selected) < batch_size and drop_last:
            break
        samples = [dataset[int(index)] for index in selected]
        yield batch_to_jittor(sparse_collate(samples))
