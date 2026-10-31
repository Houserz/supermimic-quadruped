"""Training disturbances (plan D22): large root velocity kicks that knock the robot over.

The kick is applied between two control steps, so the transition it lands in is not a function
of (state, action, xi). It is flagged for the replay recorder (excluded from world-model windows)
and for the command (the episode no longer updates the reset sampler).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def action_delay(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    lag_range: tuple[int, int] = (0, 2),
    entity_name: str = "robot",
) -> None:
    """Reset event: draw one command lag (physics substeps, uniform in ``lag_range``) per environment and
    pin it for the episode with ``Actuator.set_lags`` (plan decision F, paper Table A.3 "xi per episode").

    Runs after ``scene.reset`` (which zeroes the fused DelayBuffer's lags for these envs) and before the
    replay recorder's post-reset read, so the episode's first stored xi already carries the lag that the
    actuators will apply at every substep. The actuator cfg must use ``delay_hold_prob=1.0`` so the
    buffer never resamples on its own."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return
    lo, hi = int(lag_range[0]), int(lag_range[1])
    lags = torch.randint(lo, hi + 1, (env_ids.numel(),), device=env.device, dtype=torch.long)
    done = set()
    for act in env.scene[entity_name].actuators:
        buf = getattr(act, "_delay_buffer", None)
        if buf is not None and id(buf) not in done:  # actuators of one fused group share a buffer
            act.set_lags(lags, env_ids)
            done.add(id(buf))


def big_push(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    speed_range: tuple[float, float] = (4.0, 6.0),
    ang_speed: float = 3.0,
    down_prob: float = 0.3,
    down_speed_range: tuple[float, float] = (1.5, 3.0),
    command_name: str = "motion",
    recorder_name: str = "replay",
) -> None:
    """Add to the root velocity: a horizontal kick of random direction and magnitude in ``speed_range``
    (m/s), a roll/pitch kick in [-ang_speed, ang_speed] (rad/s), and with probability ``down_prob`` a
    downward kick in ``down_speed_range`` (m/s)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    dev = env.device
    robot = env.scene["robot"]
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev)  # noqa: E731
    vel = robot.data.root_link_vel_w[env_ids].clone()
    ang = u(0.0, 2 * math.pi)
    speed = u(*speed_range)
    vel[:, 0] += speed * torch.cos(ang)
    vel[:, 1] += speed * torch.sin(ang)
    vel[:, 2] -= torch.where(torch.rand(n, device=dev) < down_prob, u(*down_speed_range), torch.zeros(n, device=dev))
    vel[:, 3] += u(-ang_speed, ang_speed)
    vel[:, 4] += u(-ang_speed, ang_speed)
    robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    env.command_manager.get_term(command_name).pushed[env_ids] = True
    if recorder_name in getattr(env.recorder_manager, "active_terms", ()):
        env.recorder_manager.get_term(recorder_name).pending_cut[env_ids] = True
