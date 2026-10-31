"""Planner frames -> the tracker's goal features (torch), so a plan can stand in for a library clip.

The plan's local features (``features.py``) are integrated to a pose with a relative yaw (the goal
features are local, so the yaw offset is irrelevant), world velocities and body angular velocity
are re-derived the way the motion library's clips have them (root, joint and angular velocity by
forward differences, as ``mj_differentiatePos`` between consecutive frames), and
``motion_library.goal_features`` does the rest, so the ordering and definitions cannot drift from
what the tracker was trained on.
"""

from __future__ import annotations

import torch

from supermimic.envs.mdp.motion_library import RefState, goal_features
from supermimic.kinematics.rotations import rotvec_from_quat


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, u = q[..., :1], q[..., 1:]
    t = 2 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def _quat_from_zyx(yaw, pitch, roll):
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    return torch.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], -1)


def _forward_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    """(B, T, ...) forward difference along T; the last frame repeats the previous one's."""
    d = (x[:, 1:] - x[:, :-1]) / dt
    return torch.cat([d, d[:, -1:]], 1)


def plan_ref_state(feat: torch.Tensor, home: torch.Tensor, dt: float = 0.02) -> RefState:
    """(B, T, nj+8) plan features -> RefState with (B*T, ...) fields (relative yaw, xy at the origin)."""
    B, T, _ = feat.shape
    nj = home.shape[-1]
    joint = feat[..., :nj] + home
    height = feat[..., nj : nj + 1]
    g = feat[..., nj + 1 : nj + 4]
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    lin_h = feat[..., nj + 4 : nj + 7]
    wz = feat[..., nj + 7]
    yaw = torch.cat([torch.zeros_like(wz[:, :1]), torch.cumsum(0.5 * (wz[:, 1:] + wz[:, :-1]) * dt, 1)], 1)
    pitch = torch.asin(g[..., 0].clamp(-1.0, 1.0))
    roll = torch.atan2(-g[..., 1], -g[..., 2])
    quat = _quat_from_zyx(yaw, pitch, roll)
    c, s = torch.cos(yaw), torch.sin(yaw)
    vel_w = torch.stack([c * lin_h[..., 0] - s * lin_h[..., 1], s * lin_h[..., 0] + c * lin_h[..., 1]], -1)
    xy = torch.cat([torch.zeros_like(vel_w[:, :1]), torch.cumsum(0.5 * (vel_w[:, 1:] + vel_w[:, :-1]) * dt, 1)], 1)
    pos = torch.cat([xy, height], -1)
    lin_w = _forward_diff(pos, dt)  # the clips' root velocity is a forward difference, the features' a central one
    dq = _quat_mul(_quat_conj(quat[:, :-1]), quat[:, 1:])
    ang_b = rotvec_from_quat(dq) / dt
    ang_b = torch.cat([ang_b, ang_b[:, -1:]], 1)
    ang_w = _quat_apply(quat, ang_b)
    joint_vel = _forward_diff(joint, dt)
    flat = lambda x: x.reshape(B * T, *x.shape[2:])
    return RefState(root_pos_w=flat(pos), root_quat_w=flat(quat), joint_pos=flat(joint), root_lin_vel_w=flat(lin_w), root_ang_vel_w=flat(ang_w), joint_vel=flat(joint_vel))


def plan_goal_frames(feat: torch.Tensor, home: torch.Tensor, default_joint_pos: torch.Tensor, dt: float = 0.02) -> torch.Tensor:
    """(B, T, nj+8) plan features -> (B, T, 2nj+10) goal features per frame."""
    B, T, _ = feat.shape
    return goal_features(plan_ref_state(feat, home, dt), default_joint_pos).reshape(B, T, -1)
