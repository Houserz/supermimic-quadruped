"""Analytic heads of the world model (plan D4): observation, reward, termination.

Everything here is a pure differentiable function of the (13 + 2nj)-d simulator state
``s = [qpos | qvel]`` (root position expressed relative to the env origin, see ``StateIO``),
the previous clean action, the sim context ``xi`` and the reference frame, so the same code
serves the simulator-side tests and the imagined rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.envs.mdp.rewards import TorqueModel, TrackingRewardCfg, analytic_actuator_torque, tracking_reward_terms
from supermimic.kinematics.robot_fk import RobotKinematics
from supermimic.kinematics.rotations import projected_gravity, quat_apply_inverse

# Upstream G1 layout (kept for existing callers / tests).
OBS_DIM = 93
OBS_TERMS = (("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29))
ANCHOR_OBS_DIM = 9


def obs_terms_for(nj: int) -> tuple[tuple[str, int], ...]:
    return (("joint_pos", nj), ("joint_vel", nj), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", nj))


def nj_of_state(s: torch.Tensor) -> int:
    return (s.shape[-1] - 13) // 2


def split_state(s: torch.Tensor, nj: int | None = None):
    """-> root_pos (…,3), root_quat (…,4), joint_pos (…,nj), root_lin_vel_w (…,3), root_ang_vel_b (…,3), joint_vel (…,nj)."""
    nj = nj_of_state(s) if nj is None else nj
    return s[..., 0:3], s[..., 3:7], s[..., 7 : 7 + nj], s[..., 7 + nj : 10 + nj], s[..., 10 + nj : 13 + nj], s[..., 13 + nj : 13 + 2 * nj]


def analytic_observation(s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor, default_joint_pos: torch.Tensor) -> torch.Tensor:
    """Noise-free (3nj + 6)-d proprioceptive observation (paper Table A.1) from state + context.

    xi[..., 1:1+nj] is the encoder bias: observed joint positions are ``q + bias - q0``.
    """
    nj = default_joint_pos.shape[-1]
    _, root_quat, q, _, ang_b, qd = split_state(s, nj)
    bias = xi[..., 1 : 1 + nj]
    return torch.cat([q + bias - default_joint_pos, qd, ang_b, projected_gravity(root_quat), prev_action], dim=-1)


def anchor_relative_obs(robot_anchor_pos: torch.Tensor, robot_anchor_quat: torch.Tensor, ref_anchor_pos: torch.Tensor, ref_anchor_quat: torch.Tensor) -> torch.Tensor:
    """Reference anchor pose expressed in the robot anchor frame: [pos (3), rot6d (6)] (= mjlab's
    motion_anchor_pos_b / motion_anchor_ori_b, plan D5 alternative)."""
    from supermimic.kinematics.rotations import matrix_from_quat, quat_inv, quat_mul

    q10 = quat_inv(robot_anchor_quat)
    pos = quat_apply_inverse(robot_anchor_quat, ref_anchor_pos - robot_anchor_pos)
    q12 = quat_mul(q10, ref_anchor_quat)
    mat = matrix_from_quat(q12)
    return torch.cat([pos, mat[..., :2].reshape(*q12.shape[:-1], 6)], dim=-1)


def flatten_history_term_major(hist: torch.Tensor, obs_terms: tuple[tuple[str, int], ...] | None = None) -> torch.Tensor:
    """(H, N, D) oldest→newest  ->  (N, H*D) in mjlab's term-major layout [A_t0..A_tH-1, B_t0..]."""
    H, N, D = hist.shape
    if obs_terms is None:
        obs_terms = obs_terms_for((D - 6) // 3)
    chunks = []
    start = 0
    for _, d in obs_terms:
        chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(N, H * d))
        start += d
    assert start == D, (start, D)
    return torch.cat(chunks, dim=-1)


@dataclass
class ReferenceFrame:
    """Reference quantities of one target frame for a batch (N, ·)."""

    body_pos_w: torch.Tensor  # (N, B, 3)
    body_quat_w: torch.Tensor  # (N, B, 4)
    body_lin_vel_w: torch.Tensor  # (N, B, 3)
    body_ang_vel_w: torch.Tensor  # (N, B, 3)
    root_pos_w: torch.Tensor  # (N, 3)
    root_quat_w: torch.Tensor  # (N, 4)
    joint_pos: torch.Tensor | None = None  # (N, nj) for the joint-space reward terms
    joint_vel: torch.Tensor | None = None  # (N, nj)


class AnalyticHeads:
    """Reward / termination features computed from the predicted state through differentiable FK."""

    def __init__(
        self,
        fk: RobotKinematics,
        tracked_body_names: tuple[str, ...],
        anchor_body_name: str,
        root_body_name: str,
        soft_limits: torch.Tensor,
        default_joint_pos: torch.Tensor,
        reward_cfg: TrackingRewardCfg | None = None,
        support_feet: tuple[str, ...] = (),
        support_foot_threshold: float = 0.0,
        height_threshold: float = 0.25,
        tilt_threshold: float = 0.8,
        obs_noise_half: torch.Tensor | None = None,
        torque_model: "TorqueModel | None" = None,
    ):
        self.fk = fk
        self.torque_model = torque_model  # analytic actuator torque for the optional torque_headroom reward
        self.nj = fk.nj
        self.nq = fk.nq
        self.tracked_idx = fk.body_indices(tracked_body_names)
        self.anchor_index = tracked_body_names.index(anchor_body_name)
        self.root_fk_index = fk.body_index(root_body_name)
        self.soft_limits = soft_limits
        self.default_joint_pos = default_joint_pos
        self.reward_cfg = reward_cfg or TrackingRewardCfg()
        # Support-foot termination (optional): indices into the *tracked* body list.
        self.support_idx = torch.tensor([tracked_body_names.index(n) for n in support_feet], device=fk.device, dtype=torch.long) if support_feet else None
        self.support_foot_threshold = float(support_foot_threshold)
        self.height_threshold = float(height_threshold)
        self.tilt_threshold = float(tilt_threshold)
        self.num_termination_features = 2 + (1 if self.support_idx is not None else 0)
        self.obs_terms = obs_terms_for(self.nj)
        self.obs_dim = 3 * self.nj + 6
        self.obs_noise_half = obs_noise_half

    def robot_bodies(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.fk(s[..., : self.nq], s[..., self.nq :])
        idx = self.tracked_idx
        return {
            "pos": out["pos"][..., idx, :],
            "quat": out["quat"][..., idx, :],
            "lin_vel": out["lin_vel"][..., idx, :],
            "ang_vel": out["ang_vel"][..., idx, :],
            "root_pos": out["pos"][..., self.root_fk_index, :],
            "root_quat": out["quat"][..., self.root_fk_index, :],
        }

    def observation(self, s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        return analytic_observation(s, prev_action, xi, self.default_joint_pos)

    def reward(self, s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, ref: ReferenceFrame, bodies=None, xi: torch.Tensor | None = None):
        """Reward r_{t+1} of reaching s_next (= s_{t+1}) under target frame g_t. Returns dict with 'total'.

        ``xi`` (sim context) is optional: with it the torque proxy subtracts the encoder bias like the
        simulator's action term does; without it the proxy is off by at most kp * bias."""
        b = bodies if bodies is not None else self.robot_bodies(s_next)
        nj = self.nj
        torque = None
        if self.torque_model is not None and self.reward_cfg.w_torque_headroom != 0.0:
            bias = xi[..., 1 : 1 + nj] if xi is not None else None
            torque = analytic_actuator_torque(s_next[..., 7 : 7 + nj], s_next[..., 13 + nj : 13 + 2 * nj], action, self.torque_model, bias)
        return tracking_reward_terms(
            torque=torque,
            robot_body_pos_w=b["pos"],
            robot_body_quat_w=b["quat"],
            robot_body_lin_vel_w=b["lin_vel"],
            robot_body_ang_vel_w=b["ang_vel"],
            ref_body_pos_w=ref.body_pos_w,
            ref_body_quat_w=ref.body_quat_w,
            ref_body_lin_vel_w=ref.body_lin_vel_w,
            ref_body_ang_vel_w=ref.body_ang_vel_w,
            anchor_index=self.anchor_index,
            action=action,
            prev_action=prev_action,
            joint_pos=s_next[..., 7 : 7 + nj],
            joint_vel=s_next[..., 13 + nj : 13 + 2 * nj],
            ref_joint_pos=ref.joint_pos,
            ref_joint_vel=ref.joint_vel,
            soft_limits=self.soft_limits,
            cfg=self.reward_cfg,
        )

    @staticmethod
    def termination_features(root_pos: torch.Tensor, root_quat: torch.Tensor, ref: ReferenceFrame, foot_pos: torch.Tensor | None = None, ref_foot_pos: torch.Tensor | None = None) -> torch.Tensor:
        """[height error, tilt error (, max support-foot z error)] (N, 2 | 3) — the quantities the
        terminations threshold (paper 0.25 m / 0.8; Rover 0.15 m / 0.5)."""
        g = torch.tensor([0.0, 0.0, -1.0], dtype=root_quat.dtype, device=root_quat.device).expand(root_quat.shape[:-1] + (3,))
        tilt = quat_apply_inverse(ref.root_quat_w, g)[..., 2] - quat_apply_inverse(root_quat, g)[..., 2]
        height = ref.root_pos_w[..., 2] - root_pos[..., 2]
        feats = [height, tilt]
        if foot_pos is not None:
            feats.append((ref_foot_pos[..., 2] - foot_pos[..., 2]).abs().amax(-1))
        return torch.stack(feats, dim=-1)

    def termination_features_of(self, s: torch.Tensor, ref: ReferenceFrame, bodies: dict | None = None) -> torch.Tensor:
        """Termination features of state ``s`` (root pose straight from the state; feet through FK when configured)."""
        if self.support_idx is None:
            return self.termination_features(s[..., 0:3], s[..., 3:7], ref)
        b = bodies if bodies is not None else self.robot_bodies(s)
        return self.termination_features(s[..., 0:3], s[..., 3:7], ref, b["pos"][..., self.support_idx, :], ref.body_pos_w[..., self.support_idx, :])

    def terminated(self, s: torch.Tensor, ref: ReferenceFrame, height_thr: float | None = None, tilt_thr: float | None = None, bodies: dict | None = None) -> torch.Tensor:
        height_thr = self.height_threshold if height_thr is None else height_thr
        tilt_thr = self.tilt_threshold if tilt_thr is None else tilt_thr
        f = self.termination_features_of(s, ref, bodies)
        out = (f[..., 0].abs() > height_thr) | (f[..., 1].abs() > tilt_thr)
        if f.shape[-1] > 2:
            out = out | (f[..., 2] > self.support_foot_threshold)
        return out

    @staticmethod
    def root_error_features(s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Critic-only privileged drift features (plan D10): root position error in the reference
        heading frame (3) and yaw error (1). (N, 4)"""
        from supermimic.kinematics.rotations import heading_yaw, quat_apply_inverse, wrap_to_pi, yaw_quat

        dp = quat_apply_inverse(yaw_quat(ref.root_quat_w), s[..., 0:3] - ref.root_pos_w)
        dyaw = wrap_to_pi(heading_yaw(s[..., 3:7]) - heading_yaw(ref.root_quat_w))
        return torch.cat([dp, dyaw[..., None]], dim=-1)
