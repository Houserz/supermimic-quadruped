"""Per-environment simulation context xi (plan D8 / Table A.3).

xi = [foot friction (1), encoder bias (nj), COM offset of ``com_body`` (3)] and, when the robot's
actuators carry a randomised command delay (``dr.action_delay_substeps``, off on the Rover), the current lag in physics
substeps (1). nj + 4 (+1) entries; 33 for the G1 as upstream.
"""

from __future__ import annotations

import numpy as np
import torch

XI_DIM = 33


class SimContext:
    def __init__(self, env, foot_geom_regex: str, com_body_name: str = "torso_link", entity_name: str = "robot", action_delay: bool = False):
        self.env = env
        self.robot = env.scene[entity_name]
        local_ids, names = self.robot.find_geoms(foot_geom_regex)
        if len(local_ids) == 0:
            raise ValueError(f"no geoms match {foot_geom_regex}")
        self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item())
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]), dtype=torch.float32, device=env.device
        )
        self.nj = int(self.robot.data.encoder_bias.shape[1])
        self.action_delay = bool(action_delay)
        self.xi_dim = 1 + self.nj + 3 + (1 if self.action_delay else 0)
        self._delay_buffer = None
        if self.action_delay:
            for act in self.robot.actuators:
                buf = getattr(act, "_delay_buffer", None)
                if buf is not None:
                    self._delay_buffer = buf
                    break
            if self._delay_buffer is None:
                raise ValueError("action_delay requested but no actuator carries a delay buffer")

    def read(self) -> torch.Tensor:
        model = self.env.sim.model
        n = self.env.num_envs
        fr = model.geom_friction
        if fr.dim() == 3:  # expanded per world
            friction = fr[:, self.foot_geom_global, 0]
        else:
            friction = fr[self.foot_geom_global, 0].expand(n)
        ipos = model.body_ipos
        if ipos.dim() == 3:
            com = ipos[:, self.com_body_global] - self.com_nominal
        else:
            com = (ipos[self.com_body_global] - self.com_nominal).expand(n, 3)
        bias = self.robot.data.encoder_bias
        parts = [friction.reshape(n, 1), bias, com]
        if self.action_delay:
            parts.append(self._delay_buffer.current_lags.to(torch.float32).reshape(n, 1))
        xi = torch.cat(parts, dim=-1)
        assert xi.shape == (n, self.xi_dim), (xi.shape, self.xi_dim)
        return xi.clone()
