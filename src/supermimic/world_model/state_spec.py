"""State representation for the world model (plan D1–D3), robot-generic.

Simulator state  s = [root_pos(3), root_quat wxyz(4), joint_pos(nj), root_lin_vel_w(3),
                      root_ang_vel_b(3), joint_vel(nj)]            (13 + 2nj, MuJoCo layout, origin-free)

Canonical features (network input, heading-invariant, 13 + 2nj):
    [root_z(1), rot6d(heading^-1 ⊗ root_quat)(6), root_lin_vel in heading frame(3),
     root_ang_vel_b(3), joint_pos(nj), joint_vel(nj)]

Network output ``delta`` (12 + 2nj) and the integration step (dt = 0.02 s):
    dv_h(3), dw_b(3), dqd(nj)            velocity increments, heading / body frame
    dp_h(3), dth_b(3), dq(nj)            position corrections on top of explicit integration
    v'  = v + R_h dv_h               w' = w + dw_b          qd' = qd + dqd
    p'  = p + dt v' + R_h dp_h       q' = q ⊗ exp(dt w' + dth_b)    qj' = qj + dt qd' + dq
With zero output the model extrapolates at constant velocity (a physical prior); the
last layer of the dynamics MLP is zero-initialised so training starts from it.

Loss / error normalisation (D3): per-quantity nominal scales sigma applied to the
(12 + 2nj)-d error vector [dp(3), rotvec(3), dv(3), dw(3), dq(nj), dqd(nj)]. The scales are
a per-robot decision (``robots/*.py``); the G1 defaults below are the upstream values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)

# Upstream G1 constants (kept for callers that still use them).
NJ = 29
STATE_DIM = 71
FEAT_DIM = 71
DELTA_DIM = 70


def _g1_xi_offset():
    return (0.75,) + (0.0,) * 29 + (0.0,) * 3


def _g1_xi_scale():
    return (0.45,) + (0.01,) * 29 + (0.05,) * 3


@dataclass
class StateSpecCfg:
    nj: int = NJ
    dt: float = 0.02
    # Nominal error scales (D3) for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics head for [dv_h, dw_b, dqd, dp_h, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel] (divide by these).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    # xi normalisation: (offset, scale) per xi entry: friction, encoder bias (nj), com offset (3) [, action delay].
    xi_offset: tuple[float, ...] = field(default_factory=_g1_xi_offset)
    xi_scale: tuple[float, ...] = field(default_factory=_g1_xi_scale)

    @staticmethod
    def from_robot(spec, dt: float = 0.02, explicit_integration: bool = True) -> "StateSpecCfg":
        """Scales from a ``RobotSpec`` (``robots/spec.py``)."""
        s = spec.state_scales
        nj = spec.nj
        xi_off = (s.xi_friction[0],) + (s.xi_bias[0],) * nj + (s.xi_com[0],) * 3
        xi_sc = (s.xi_friction[1],) + (s.xi_bias[1],) * nj + (s.xi_com[1],) * 3
        if spec.has_action_delay:
            xi_off += (s.xi_delay[0],)
            xi_sc += (s.xi_delay[1],)
        return StateSpecCfg(nj=nj, dt=dt, err_scale=s.err_scale, out_scale=s.out_scale, in_scale=s.in_scale, explicit_integration=explicit_integration, xi_offset=xi_off, xi_scale=xi_sc)

    @property
    def state_dim(self) -> int:
        return 13 + 2 * self.nj

    @property
    def delta_dim(self) -> int:
        return 12 + 2 * self.nj

    @property
    def xi_dim(self) -> int:
        return len(self.xi_offset)


class StateSpec:
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32):
        self.cfg = cfg
        self.dt = cfg.dt
        self.nj = int(cfg.nj)
        nj = self.nj
        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)])  # noqa: E731
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, nj, nj)).to(device=device, dtype=dtype)
        self.out_scale = rep(cfg.out_scale, (3, 3, nj, 3, 3, nj)).to(device=device, dtype=dtype)
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, nj, nj)).to(device=device, dtype=dtype)
        self.xi_offset = torch.tensor(cfg.xi_offset, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(cfg.xi_scale, device=device, dtype=dtype)
        self.state_dim = cfg.state_dim
        self.delta_dim = cfg.delta_dim
        self.feat_dim = cfg.state_dim
        self.xi_dim = cfg.xi_dim

    # ------------------------------------------------------------------ split / features

    def split(self, s: torch.Tensor):
        nj = self.nj
        return s[..., 0:3], s[..., 3:7], s[..., 7 : 7 + nj], s[..., 7 + nj : 10 + nj], s[..., 10 + nj : 13 + nj], s[..., 13 + nj : 13 + 2 * nj]

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Heading-invariant, scaled network input features (…, 13 + 2nj)."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
        v_h = quat_apply_inverse(hq, v)
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1)
        return feat / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the (unscaled) network output ``delta`` (…, 12 + 2nj) to state ``s``."""
        nj = self.nj
        d = delta * self.out_scale
        dv_h, dw_b, dqd = d[..., 0:3], d[..., 3:6], d[..., 6 : 6 + nj]
        dp_h, dth_b, dq = d[..., 6 + nj : 9 + nj], d[..., 9 + nj : 12 + nj], d[..., 12 + nj : 12 + 2 * nj]
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        dt = self.dt if self.cfg.explicit_integration else 0.0
        p_new = p + dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dth_b)))
        qj_new = qj + dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised (12 + 2nj)-d error [dp, rotvec, dv, dw, dq, dqd] / sigma."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
        rot_err = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q2), q1))
        err = torch.cat([p1 - p2, rot_err, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1)
        return err / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
