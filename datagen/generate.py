import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


HEX8 = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.int32,
)


def get_node_type(voxel: np.ndarray) -> np.ndarray:
    res = voxel.shape[0]
    node_type = np.zeros((res, res, res, 8), dtype=np.uint8)
    solid = np.argwhere(voxel)
    for corner, offset in enumerate(HEX8):
        coord = (solid + offset) % res
        node_type[coord[:, 0], coord[:, 1], coord[:, 2], corner] = 1
    return node_type


def load_voxel_from_file(path: Path, res: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    elif path.suffix.lower() == ".csv":
        array = np.loadtxt(path, delimiter=",")
    elif path.suffix.lower() == ".npz":
        array = np.load(path)["voxel"]
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return (np.asarray(array).reshape(res, res, res) > 0)


def process_one_voxel(voxel: np.ndarray, res: int):
    voxel = voxel.reshape(res, res, res).astype(bool, copy=False)
    voxel_coord = np.argwhere(voxel).astype(np.int32)
    node_grid = np.zeros_like(voxel)
    for offset in HEX8:
        coord = (voxel_coord + offset) % res
        node_grid[coord[:, 0], coord[:, 1], coord[:, 2]] = True
    node_coord = np.argwhere(node_grid).astype(np.int32)

    node_type_grid = get_node_type(voxel)
    node_type = node_type_grid[node_coord[:, 0], node_coord[:, 1], node_coord[:, 2]]
    node_id_grid = np.zeros_like(voxel, dtype=np.int32)
    node_id_grid[node_coord[:, 0], node_coord[:, 1], node_coord[:, 2]] = np.arange(
        node_coord.shape[0], dtype=np.int32
    )

    node_index = np.zeros((voxel_coord.shape[0], 8), dtype=np.int32)
    for corner, offset in enumerate(HEX8):
        coord = (voxel_coord + offset) % res
        node_index[:, corner] = node_id_grid[coord[:, 0], coord[:, 1], coord[:, 2]]
    return node_coord, node_type, node_index, voxel


def unique_path_if_exists(path: Path) -> Path:
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def folder_id(index: int) -> str:
    result = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--type", type=str, default="Truss")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--avoid_collision", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = skipped = 0
    for dir_index, directory in enumerate(args.input_dirs):
        directory = Path(directory)
        if not directory.exists():
            print(f"[WARN] input dir not found: {directory}")
            continue
        files = [
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".npy", ".csv", ".npz"}
        ]
        for path in tqdm(files, desc=f"Processing {directory.name}", leave=False):
            out_path = out_dir / f"{args.type}_{folder_id(dir_index)}_{path.stem}.npz"
            if out_path.exists() and not args.overwrite and not args.avoid_collision:
                skipped += 1
                continue
            if args.avoid_collision and not args.overwrite:
                out_path = unique_path_if_exists(out_path)
            try:
                coords, node_type, node_index, voxel = process_one_voxel(
                    load_voxel_from_file(path, args.res), args.res
                )
                np.savez(
                    out_path,
                    coords=coords,
                    node_type=node_type,
                    voxel=voxel,
                    node_index=node_index,
                )
                processed += 1
            except Exception as error:
                print(f"[ERROR] failed on {path}: {error}")
    print(f"Done. Processed={processed}, Skipped={skipped}, Out={out_dir}")


if __name__ == "__main__":
    main()
