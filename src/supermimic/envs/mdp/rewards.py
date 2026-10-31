"""Analytic BeyondMimic tracking reward in torch (plan D4/D7), plus optional joint-space terms.

One differentiable implementation shared by the simulator side (tests / logging)
and the imagination side (world-model reward head). Mirrors mjlab's
``tasks/tracking/mdp/rewards.py`` term by term, including the re-anchoring of the
reference bodies to the robot anchor in xy + yaw only
(``MotionCommand.update_relative_body_poses``).

Inputs are world-frame link quantities of the *tracked* bodies (same ordering as
the robot spec's ``tracked_bodies``), the anchor being one of them (``anchor_index``).
Rewards are returned already weighted and multiplied by ``dt`` like mjlab's reward
manager (``scale_rewards_by_dt=True``).

Joint-space tracking (``motion_joint_pos`` / ``motion_joint_vel``, optional):
``exp(-mean_sq_error / std^2)`` over all joints; weight 0 on the G1 (paper recipe) and the Rover.
Useful on robots with many joints, where many configurations put the tracked bodies in nearly
the same poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_box_minus, quat_inv, quat_mul, yaw_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass
class TrackingRewardCfg:
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    std_joint_pos: float = 1.0
    std_joint_vel: float = 1.0
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_joint_pos: float = 0.0
    w_joint_vel: float = 0.0
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    w_torque_headroom: float = 0.0  # optional: weight on sum_j relu(|tau_j| - threshold)^2; 0 = term off
    torque_threshold: float = 0.0  # N*m, the actuator's continuous rating
    dt: float = 0.02

    @staticmethod
    def from_robot(spec, dt: float = 0.02) -> "TrackingRewardCfg":
        return TrackingRewardCfg(
            w_torque_headroom=spec.reward_w_torque_headroom,
            torque_threshold=spec.torque_threshold,
            std_body_pos=spec.reward_std_body_pos,
            std_body_ori=spec.reward_std_body_ori,
            std_body_lin_vel=spec.reward_std_body_lin_vel,
            std_body_ang_vel=spec.reward_std_body_ang_vel,
            std_root_pos=spec.reward_std_root_pos,
            std_root_ori=spec.reward_std_root_ori,
            std_joint_pos=spec.reward_std_joint_pos,
            std_joint_vel=spec.reward_std_joint_vel,
            w_joint_pos=spec.reward_w_joint_pos,
            w_joint_vel=spec.reward_w_joint_vel,
            w_action_rate=spec.reward_w_action_rate,
            w_joint_limit=spec.reward_w_joint_limit,
            dt=dt,
        )

    @property
    def max_step_reward(self) -> float:
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori + self.w_joint_pos + self.w_joint_vel)

    @property
    def term_names(self) -> tuple[str, ...]:
        names = ["motion_global_root_pos", "motion_global_root_ori", "motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel"]
        if self.w_joint_pos != 0.0:
            names.append("motion_joint_pos")
        if self.w_joint_vel != 0.0:
            names.append("motion_joint_vel")
        names += ["action_rate_l2", "joint_limit"]
        if self.w_torque_headroom != 0.0:
            names.append("torque_headroom")
        return tuple(names)


# ---------------------------------------------------------------- torque headroom (optional)


@dataclass
class TorqueModel:
    """Analytic actuator torque for the imagination side: a PD servo clipped to its torque–speed
    curve, as tensors (for robots whose spec sets ``torque_model``; none of the shipped ones does)."""

    kp: float
    kd: float
    action_scale: torch.Tensor  # (nj,) or scalar
    default_joint_pos: torch.Tensor  # (nj,)
    effort_limit: float
    plateau_torque: float
    corner_speed: float
    decay_speed: float
    cutoff_speed: float


def analytic_actuator_torque(joint_pos: torch.Tensor, joint_vel: torch.Tensor, action: torch.Tensor, tm: TorqueModel, encoder_bias: torch.Tensor | None = None) -> torch.Tensor:
    """tau = clip(kp (q0 + scale a - bias - q) - kd qd, +-min(curve(qd), effort_limit)), (N, nj).

    The simulator evaluates the same law at every physics substep on the *delayed* target and the
    state at that substep; this proxy uses the step's clean action and the end-of-step state (the
    quantities the world model predicts). ``encoder_bias`` (xi) is subtracted from the target like
    mjlab's action term does; without it the proxy is off by at most kp * bias (0.1 N*m here)."""
    target = tm.default_joint_pos + tm.action_scale * action
    if encoder_bias is not None:
        target = target - encoder_bias
    torque = tm.kp * (target - joint_pos) - tm.kd * joint_vel
    w = joint_vel.abs()
    limit = tm.plateau_torque * torch.exp(-(w - tm.corner_speed).clamp(min=0.0) / tm.decay_speed)
    limit = torch.where(w >= tm.cutoff_speed, torch.zeros_like(limit), limit).clamp(max=tm.effort_limit)
    return torch.clamp(torque, -limit, limit)


def torque_headroom(torque: torch.Tensor, threshold: float) -> torch.Tensor:
    """sum_j relu(|tau_j| - threshold)^2 (N,): zero below the continuous rating, quadratic above."""
    return ((torque.abs() - threshold).clamp(min=0.0) ** 2).sum(-1)


def reanchor_reference(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express the reference bodies in a frame that shares the robot anchor's xy position and yaw.

    Exactly mjlab's ``update_relative_body_poses``: z, roll and pitch stay those of the reference.
    Shapes: bodies (N, B, ·), anchors (N, ·).
    """
    nb = ref_body_pos_w.shape[-2]
    delta_pos = robot_anchor_pos_w.clone()
    delta_pos = torch.cat([delta_pos[..., :2], ref_anchor_pos_w[..., 2:3]], dim=-1)  # (N,3)
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))  # (N,4)
    delta_ori_b = delta_ori[..., None, :].expand(*delta_ori.shape[:-1], nb, 4)
    rel_quat = quat_mul(delta_ori_b, ref_body_quat_w)
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori_b, ref_body_pos_w - ref_anchor_pos_w[..., None, :])
    return rel_pos, rel_quat


def _sq_quat_err(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between quaternions, smooth at zero (== quat_error_magnitude**2)."""
    v = quat_box_minus(q1, q2)
    return (v * v).sum(-1)


def tracking_reward_terms(
    *,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    anchor_index: int,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,  # (nj, 2) or (N, nj, 2)
    cfg: TrackingRewardCfg,
    joint_vel: torch.Tensor | None = None,
    ref_joint_pos: torch.Tensor | None = None,
    ref_joint_vel: torch.Tensor | None = None,
    torque: torch.Tensor | None = None,  # (N, nj) actuator torque of the step, for the torque_headroom term
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (N,) plus 'total'."""
    ra_pos = robot_body_pos_w[..., anchor_index, :]
    ra_quat = robot_body_quat_w[..., anchor_index, :]
    ma_pos = ref_body_pos_w[..., anchor_index, :]
    ma_quat = ref_body_quat_w[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)
    e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)
    e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w).mean(-1)
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)

    lo = soft_limits[..., 0]
    hi = soft_limits[..., 1]
    out = (-(joint_pos - lo)).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)
    joint_limit = out.sum(-1)
    action_rate = ((action - prev_action) ** 2).sum(-1)

    dt = cfg.dt
    terms = {
        "motion_global_root_pos": dt * cfg.w_root_pos * torch.exp(-e_root_pos / cfg.std_root_pos**2),
        "motion_global_root_ori": dt * cfg.w_root_ori * torch.exp(-e_root_ori / cfg.std_root_ori**2),
        "motion_body_pos": dt * cfg.w_body_pos * torch.exp(-e_body_pos / cfg.std_body_pos**2),
        "motion_body_ori": dt * cfg.w_body_ori * torch.exp(-e_body_ori / cfg.std_body_ori**2),
        "motion_body_lin_vel": dt * cfg.w_body_lin_vel * torch.exp(-e_body_lin / cfg.std_body_lin_vel**2),
        "motion_body_ang_vel": dt * cfg.w_body_ang_vel * torch.exp(-e_body_ang / cfg.std_body_ang_vel**2),
    }
    if cfg.w_joint_pos != 0.0:
        assert ref_joint_pos is not None, "joint-space reward needs the reference joint positions"
        e_q = ((ref_joint_pos - joint_pos) ** 2).mean(-1)
        terms["motion_joint_pos"] = dt * cfg.w_joint_pos * torch.exp(-e_q / cfg.std_joint_pos**2)
    if cfg.w_joint_vel != 0.0:
        assert ref_joint_vel is not None and joint_vel is not None, "joint-space reward needs the joint velocities"
        e_qd = ((ref_joint_vel - joint_vel) ** 2).mean(-1)
        terms["motion_joint_vel"] = dt * cfg.w_joint_vel * torch.exp(-e_qd / cfg.std_joint_vel**2)
    terms["action_rate_l2"] = dt * cfg.w_action_rate * action_rate
    terms["joint_limit"] = dt * cfg.w_joint_limit * joint_limit
    if cfg.w_torque_headroom != 0.0:
        assert torque is not None, "torque_headroom reward needs the actuator torque"
        terms["torque_headroom"] = dt * cfg.w_torque_headroom * torque_headroom(torque, cfg.torque_threshold)
    total = terms["motion_global_root_pos"]
    for k, v in terms.items():
        if k != "motion_global_root_pos":
            total = total + v
    terms["total"] = total
    return terms


# ---------------------------------------------------------------- mjlab reward terms (simulator side)


def torque_model_for(spec, entity, device) -> TorqueModel | None:
    """Analytic twin of the robot's PD + torque-curve actuator (None when the term is off): gains and
    curve from the spec, q0 from the entity, the per-joint action scale from the spec's scale rule."""
    import re

    tm = spec.torque_model
    if tm is None or spec.reward_w_torque_headroom == 0.0:
        return None
    q0 = entity.data.default_joint_pos[0].clone()
    scale = spec.action_scale
    if isinstance(scale, dict):
        scale_t = torch.tensor([float(next(v for k, v in scale.items() if re.fullmatch(k, n))) for n in entity.joint_names], device=device)
    else:
        scale_t = torch.full_like(q0, float(scale))
    return TorqueModel(kp=tm.kp, kd=tm.kd, action_scale=scale_t, default_joint_pos=q0, effort_limit=tm.effort_limit, plateau_torque=tm.plateau_torque, corner_speed=tm.corner_speed, decay_speed=tm.decay_speed, cutoff_speed=tm.cutoff_speed)


def torque_headroom_cost(env: "ManagerBasedRlEnv", asset_cfg, threshold: float, robot: str, action_name: str = "joint_pos") -> torch.Tensor:
    """Torque-headroom cost sum_j relu(|tau_j| - threshold)^2 (optional, needs ``spec.torque_model``).

    The torque is the *analytic* servo law on the end-of-step state, the executed (noisy) raw action
    and the entity's encoder bias — the same function the imagination side evaluates on the predicted
    state and the clean action (``AnalyticHeads.reward``), so the two sides agree by construction.
    It is not mjlab's ``actuator_force``: that is the PD output of the *last physics substep* (one
    substep older than the state the reward sees, and a stiff servo's standing jitter at its PD
    frequency can move it by up to ~0.8 N*m within a substep), which no state-based reward can reproduce.
    ``scripts/eval_torque_budget.py`` reports the physical ``actuator_force`` budget separately."""
    from supermimic.robots import get_robot_spec

    cache = env.__dict__.setdefault("_sm_torque_models", {})
    tm = cache.get(asset_cfg.name)
    if tm is None:
        tm = cache[asset_cfg.name] = torque_model_for(get_robot_spec(robot), env.scene[asset_cfg.name], env.device)
    asset = env.scene[asset_cfg.name]
    a_exec = env.action_manager.get_term(action_name).executed_raw
    tau = analytic_actuator_torque(asset.data.joint_pos, asset.data.joint_vel, a_exec, tm, asset.data.encoder_bias)
    return torque_headroom(tau, threshold)


def motion_joint_pos_error_exp(env: "ManagerBasedRlEnv", command_name: str, std: float) -> torch.Tensor:
    """Joint-space position tracking, ``exp(-mean_sq / std^2)`` (optional term)."""
    cmd = env.command_manager.get_term(command_name)
    error = torch.square(cmd.joint_pos - cmd.robot_joint_pos)
    return torch.exp(-error.mean(dim=-1) / std**2)


def motion_joint_vel_error_exp(env: "ManagerBasedRlEnv", command_name: str, std: float) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    error = torch.square(cmd.joint_vel - cmd.robot_joint_vel)
    return torch.exp(-error.mean(dim=-1) / std**2)
