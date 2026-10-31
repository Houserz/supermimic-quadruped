"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.heads import ReferenceFrame
from supermimic.world_model.world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w, r.joint_pos, r.joint_vel)


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        wm.enable_compile(cfg.compile)
        self.nan_skips = 0
        self.dump_path = "wm_nan_batch.pt"
        self.needs_bodies = wm.term_head.n_feat > 2  # support-foot termination feature (optional)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape
        flat = w.frame.reshape(-1)
        r = self.lib.ref(flat)
        body_pos = self.lib.bodies(flat).pos_w if self.needs_bodies else None
        refs = []
        for j in range(k):
            sl = slice(j * n, (j + 1) * n)
            refs.append(ReferenceFrame(None if body_pos is None else body_pos[sl], None, None, None, r.root_pos_w[sl], r.root_quat_w[sl]))
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs, w.immune)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        for _ in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            loss, m = self.loss_on(w)
            self.opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                self.nan_skips += 1
                if self.nan_skips == 1:
                    torch.save({"states": w.states, "a_exec": w.a_exec, "xi": w.xi, "done": w.done, "frame": w.frame}, self.dump_path)
                    print(f"[WARN] non-finite world-model loss; batch dumped to {self.dump_path}")
                continue
            loss.backward()
            allreduce_grads(self.wm)
            gn = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip)
            self.opt.step()
            m["loss"] = loss.detach()
            m["grad_norm"] = gn.detach()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        # Single host sync per update (keeps the CPU queuing GPU work across minibatches).
        out = {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}
        out["nan_skips"] = float(self.nan_skips)
        return out

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)
