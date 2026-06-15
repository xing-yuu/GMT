"""Pure-Jittor sparse point transformer used by GMT.

The original implementation relied on spconv and flash-attn. Jittor does not
provide API-compatible versions of those packages, so this module keeps sparse
point features and uses serialized window attention plus geometric pooling.
"""

import math
from typing import Dict, List, Tuple

import jittor as jt
import numpy as np
from jittor import nn

from train._utils import scatter_mean


def _serialized_padding(
    coord: jt.Var, patch_size: int, order_index: int
) -> Tuple[jt.Var, jt.Var]:
    """Return padded serialized indices and an inverse map."""
    coords = coord.numpy().astype(np.int64, copy=False)
    batch = coords[:, 0]
    xyz = coords[:, 1:4]
    axes = [(0, 1, 2), (1, 2, 0), (2, 0, 1)][order_index % 3]
    permutation = np.lexsort(
        (xyz[:, axes[2]], xyz[:, axes[1]], xyz[:, axes[0]], batch)
    )

    padded = []
    for batch_id in np.unique(batch):
        segment = permutation[batch[permutation] == batch_id]
        target = max(patch_size, int(math.ceil(len(segment) / patch_size) * patch_size))
        if len(segment) < target:
            segment = np.concatenate(
                [segment, np.repeat(segment[-1:], target - len(segment))]
            )
        padded.append(segment)
    padded = np.concatenate(padded).astype(np.int32)

    inverse = np.empty((coords.shape[0],), dtype=np.int32)
    seen = np.zeros((coords.shape[0],), dtype=bool)
    for position, original in enumerate(padded):
        if not seen[original]:
            inverse[original] = position
            seen[original] = True
    return jt.array(padded).int32(), jt.array(inverse).int32()


def _pool_geometry(point: Dict, resolution: int):
    coords = point["coord"].numpy().astype(np.int32, copy=False)
    coarse = coords.copy()
    coarse[:, 1:4] = (coarse[:, 1:4] // 2) % max(1, resolution // 2)
    unique, inverse = np.unique(coarse, axis=0, return_inverse=True)
    inverse_var = jt.array(inverse.astype(np.int32)).int32()
    feat = scatter_mean(point["feat"], inverse_var, unique.shape[0])
    return {
        "coord": jt.array(unique.astype(np.int32)).int32(),
        "feat": feat,
        "resolution": max(1, resolution // 2),
        "order_cache": {},
    }, inverse_var


class MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.act = nn.GELU()

    def execute(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SerializedAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, patch_size: int, order_index: int):
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.order_index = order_index
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def execute(self, point: Dict):
        cache_key = (self.patch_size, self.order_index)
        if cache_key not in point["order_cache"]:
            point["order_cache"][cache_key] = _serialized_padding(
                point["coord"], self.patch_size, self.order_index
            )
        padded, inverse = point["order_cache"][cache_key]

        x = point["feat"][padded]
        window_count = x.shape[0] // self.patch_size
        head_dim = self.channels // self.num_heads
        qkv = self.qkv(x).reshape(
            window_count, self.patch_size, 3, self.num_heads, head_dim
        )
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        attn = nn.softmax(jt.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale, dim=-1)
        out = jt.matmul(attn, v).transpose(0, 2, 1, 3).reshape(-1, self.channels)
        point["feat"] = self.proj(out[inverse])
        return point


class Block(nn.Module):
    def __init__(self, channels: int, num_heads: int, patch_size: int, order_index: int):
        self.norm1 = nn.LayerNorm(channels)
        self.attn = SerializedAttention(channels, num_heads, patch_size, order_index)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MLP(channels, channels * 4, channels)

    def execute(self, point: Dict):
        shortcut = point["feat"]
        point["feat"] = self.norm1(point["feat"])
        point = self.attn(point)
        point["feat"] = shortcut + point["feat"]
        point["feat"] = point["feat"] + self.mlp(self.norm2(point["feat"]))
        return point


class Stage(nn.Module):
    def __init__(self, channels: int, depth: int, heads: int, patch_size: int):
        self.blocks = nn.ModuleList(
            [
                Block(channels, heads, patch_size, block_id % 3)
                for block_id in range(depth)
            ]
        )

    def execute(self, point: Dict):
        for block in self.blocks:
            point = block(point)
        return point


class PointTransformerV3(nn.Module):
    """Multi-scale serialized point transformer with a U-Net decoder."""

    def __init__(self, cfg):
        channels = list(cfg.model.trans_channel)
        depths = list(cfg.model.trans_block)
        heads = list(cfg.model.trans_head)
        windows = list(cfg.model.trans_window_size)
        self.channels = channels
        self.num_stages = len(channels)
        assert self.num_stages == len(depths) == len(heads) == len(windows)
        assert int(cfg.resolution) % (2 ** (self.num_stages - 1)) == 0
        assert all(channel % head == 0 for channel, head in zip(channels, heads))
        assert all(window > 0 for window in windows)

        self.position = nn.ModuleList(
            [MLP(3, channel, channel) for channel in channels]
        )
        self.enc = nn.ModuleList(
            [
                Stage(channels[i], depths[i], heads[i], windows[i])
                for i in range(self.num_stages)
            ]
        )
        self.down_project = nn.ModuleList(
            [nn.Linear(channels[i], channels[i + 1]) for i in range(self.num_stages - 1)]
        )
        self.up_project = nn.ModuleList(
            [nn.Linear(channels[i + 1], channels[i]) for i in range(self.num_stages - 1)]
        )
        self.fuse = nn.ModuleList(
            [nn.Linear(channels[i] * 2, channels[i]) for i in range(self.num_stages - 1)]
        )
        self.dec = nn.ModuleList(
            [
                Stage(channels[i], depths[i], heads[i], windows[i])
                for i in range(self.num_stages - 1)
            ]
        )

    def execute(self, data_dict: Dict) -> List[Dict]:
        point = {
            "coord": data_dict["coord"],
            "feat": data_dict["feat"],
            "resolution": int(data_dict["resolution"]),
            "order_cache": {},
        }
        levels = []
        pool_inverse = []

        for stage_id in range(self.num_stages):
            coord = point["coord"][:, 1:4].float32() / float(point["resolution"])
            point["feat"] = point["feat"] + self.position[stage_id](coord)
            point = self.enc[stage_id](point)
            levels.append(point)
            if stage_id < self.num_stages - 1:
                point, inverse = _pool_geometry(point, point["resolution"])
                point["feat"] = self.down_project[stage_id](point["feat"])
                pool_inverse.append(inverse)

        decoded = [levels[-1]]
        point = levels[-1]
        for stage_id in reversed(range(self.num_stages - 1)):
            up = self.up_project[stage_id](point["feat"])[pool_inverse[stage_id]]
            point = {
                "coord": levels[stage_id]["coord"],
                "feat": self.fuse[stage_id](
                    jt.concat([up, levels[stage_id]["feat"]], dim=1)
                ),
                "resolution": levels[stage_id]["resolution"],
                "order_cache": {},
            }
            point = self.dec[stage_id](point)
            decoded.append(point)

        return list(reversed(decoded))
