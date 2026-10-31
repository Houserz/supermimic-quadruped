"""Deterministic planner: (context frames, velocity condition, gait) -> future frames.

Inputs and outputs are normalised features (``Normalizer``). The network predicts the future as a
residual on the last context frame, so a plan starts where the executing one is and the network
only has to learn the motion from there.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class PlannerCfg:
    feat_dim: int
    context: int = 25  # frames of the executing plan the planner sees (0.5 s at 50 Hz)
    horizon: int = 50  # frames it plans (1.0 s)
    num_gaits: int = 1
    hidden: int = 1024
    blocks: int = 4
    dropout: float = 0.0


class Normalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, cond_scale: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("cond_scale", cond_scale)

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean


class _Block(nn.Module):
    def __init__(self, h: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Dropout(dropout), nn.Linear(h, h))

    def forward(self, x):
        return x + self.net(x)


class Planner(nn.Module):
    def __init__(self, cfg: PlannerCfg, normalizer: Normalizer):
        super().__init__()
        self.cfg = cfg
        self.normalizer = normalizer
        d, c, t = cfg.feat_dim, cfg.context, cfg.horizon
        self.inp = nn.Linear(c * d + 3 + cfg.num_gaits, cfg.hidden)
        self.blocks = nn.Sequential(*[_Block(cfg.hidden, cfg.dropout) for _ in range(cfg.blocks)])
        self.out = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, t * d))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, ctx_n: torch.Tensor, cond: torch.Tensor, gait: torch.Tensor) -> torch.Tensor:
        """ctx_n (B, C, D) normalised, cond (B, 3) in m/s, m/s, rad/s, gait (B,) -> (B, T, D) normalised."""
        b = ctx_n.shape[0]
        g = nn.functional.one_hot(gait, self.cfg.num_gaits).float()
        x = torch.cat([ctx_n.reshape(b, -1), cond / self.normalizer.cond_scale, g], dim=-1)
        y = self.out(self.blocks(self.inp(x))).reshape(b, self.cfg.horizon, self.cfg.feat_dim)
        return ctx_n[:, -1:, :] + y

    @torch.no_grad()
    def plan(self, ctx: torch.Tensor, cond: torch.Tensor, gait: torch.Tensor) -> torch.Tensor:
        """Unnormalised in and out: ctx (B, C, D) -> (B, T, D)."""
        n = self.normalizer
        return n.denorm(self(n.norm(ctx), cond, gait))

    def save(self, path, **extra) -> None:
        n = self.normalizer
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict(), "mean": n.mean, "std": n.std, "cond_scale": n.cond_scale, **extra}, path)

    @classmethod
    def load(cls, path, device="cpu") -> tuple[Planner, dict]:
        ck = torch.load(path, map_location=device, weights_only=False)
        model = cls(PlannerCfg(**ck["cfg"]), Normalizer(ck["mean"], ck["std"], ck["cond_scale"])).to(device)
        model.load_state_dict(ck["state"])
        model.eval()
        return model, ck
