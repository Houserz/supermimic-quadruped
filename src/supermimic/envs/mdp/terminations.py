"""Paper terminations: root height deviation and tilt; clip end as time-out; optional support feet.

All failure terminations are masked for the command's ``immune`` episodes (get-up training, plan D22).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return ((cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold) & ~cmd.immune


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
    return ((ref - rob).abs() > threshold) & ~cmd.immune


def bad_support_foot_height(env: ManagerBasedRlEnv, command_name: str, threshold: float, body_names: tuple[str, ...]) -> torch.Tensor:
    """A support foot's world z deviates from the reference's by more than ``threshold``
    (z is yaw/xy-anchoring invariant, see heads.py)."""
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    idx = cmd.tracked_index_of(body_names)
    err = (cmd.body_pos_w[:, idx, 2] - cmd.robot_body_pos_w[:, idx, 2]).abs()
    return (err.amax(-1) > threshold) & ~cmd.immune


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.clip_end
