"""Morton serialization helpers implemented with Jittor integer operations."""

import jittor as jt


def z_order_encode(grid_coord: jt.Var, depth: int = 16):
    coord = grid_coord.int64()
    code = jt.zeros((coord.shape[0],), dtype="int64")
    for bit in range(depth):
        code = code | (((coord[:, 0] >> bit) & 1) << (3 * bit + 2))
        code = code | (((coord[:, 1] >> bit) & 1) << (3 * bit + 1))
        code = code | (((coord[:, 2] >> bit) & 1) << (3 * bit))
    return code


def z_order_decode(code: jt.Var, depth: int = 16):
    xyz = [jt.zeros(code.shape, dtype="int64") for _ in range(3)]
    for bit in range(depth):
        xyz[0] = xyz[0] | (((code >> (3 * bit + 2)) & 1) << bit)
        xyz[1] = xyz[1] | (((code >> (3 * bit + 1)) & 1) << bit)
        xyz[2] = xyz[2] | (((code >> (3 * bit)) & 1) << bit)
    return jt.stack(xyz, dim=1)


def encode(grid_coord, batch=None, depth=16, order="z"):
    if order == "x":
        grid_coord = grid_coord[:, [1, 2, 0]]
    elif order == "y":
        grid_coord = grid_coord[:, [2, 0, 1]]
    elif order != "z":
        raise ValueError(f"Unsupported serialization order: {order}")
    code = z_order_encode(grid_coord, depth)
    if batch is not None:
        code = (batch.int64() << (depth * 3)) | code
    return code


def decode(code, depth=16, order="z"):
    batch = code >> (depth * 3)
    coord = z_order_decode(code & ((1 << (depth * 3)) - 1), depth)
    if order == "x":
        coord = coord[:, [2, 0, 1]]
    elif order == "y":
        coord = coord[:, [1, 2, 0]]
    return coord, batch
