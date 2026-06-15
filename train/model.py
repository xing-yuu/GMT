import jittor as jt
from jittor import nn

from train.EBE_GMG import GMGSolver
from train.PTv3_3 import PointTransformerV3


class Decoder(nn.Module):
    def __init__(self, in_channels: int):
        hidden = max(16, in_channels // 2)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def execute(self, features):
        return self.net(features)


class GMT(nn.Module):
    """Jittor implementation of the GMT neural homogenization solver."""

    def __init__(self, cfg):
        self.cfg = cfg
        first_channel = int(cfg.model.trans_channel[0])
        self.input_encoder = nn.Sequential(
            nn.Linear(8, first_channel),
            nn.GELU(),
            nn.LayerNorm(first_channel),
        )
        self.ptv3 = PointTransformerV3(cfg)
        self.decoders = nn.ModuleList(
            [
                nn.ModuleList(
                    [Decoder(int(channel)) for channel in cfg.model.trans_channel]
                )
                for _ in range(6)
            ]
        )
        self.gmg_solver = GMGSolver()

    def execute(self, batch, Ke, F):
        levels = self.ptv3(
            {
                "coord": batch["coord"],
                "feat": self.input_encoder(batch["feature"]),
                "resolution": int(self.cfg.resolution),
            }
        )

        six_u = []
        six_r = []
        for equation in range(6):
            initial = [
                {
                    "coord": level["coord"],
                    "u": self.decoders[equation][level_id](level["feat"]),
                }
                for level_id, level in enumerate(levels)
            ]
            u, residual = self.gmg_solver(
                fine_ke=Ke,
                global_f=F[:, :, equation],
                node_coords_L0=batch["coord"],
                fine_topo_indices=batch["node_index"],
                max_iter=int(self.cfg.GMG.max_cycle),
                smooth_iter=list(self.cfg.GMG.smooth_iter),
                init_list=initial,
            )
            six_u.append(u.unsqueeze(-1))
            six_r.append(residual.unsqueeze(-1))

        return jt.concat(six_u, dim=2), jt.concat(six_r, dim=2)
