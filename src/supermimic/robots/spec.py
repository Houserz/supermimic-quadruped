"""Robot description shared by every module that used to hard-code the Unitree G1.

Everything dimensional in SuperMimic follows from the number of actuated hinge joints
``nj`` (plan §1.3 table); everything scale-dependent (error normalisation, reward stds,
termination thresholds, domain randomisation, noise) is a *decision* per robot and lives
in the spec with the reasoning next to it (``g1.py``, ``rover.py``).

Derived layout (MuJoCo free-joint root + ``nj`` hinges):

    state   s = [root_pos 3, root_quat 4, q nj, root_lin_vel_w 3, root_ang_vel_b 3, qd nj]   13 + 2nj
    delta   [dv_h 3, dw_b 3, dqd nj, dp_h 3, dth_b 3, dq nj]                                  12 + 2nj
    obs     [q - q0 nj, qd nj, ang_vel_b 3, projected_gravity 3, prev_action nj]             3nj + 6
    goal    [q - q0 nj, qd nj, grav 3, ang_b 3, lin_h 3, height 1]  per future frame         2nj + 10
    xi      [friction 1, encoder_bias nj, com_offset 3 (, action_delay 1)]                    nj + 4 (+1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg


@dataclass(frozen=True)
class TorqueModelSpec:
    """What the analytic torque proxy needs (imagination side of the ``torque_headroom`` reward):
    a PD servo with common gains on all joints whose output is clipped to a torque–speed curve.
    tau = clip(kp (q0 + scale a - q) - kd qd, +-min(curve(qd), effort_limit))."""

    kp: float
    kd: float
    effort_limit: float
    plateau_torque: float
    corner_speed: float
    decay_speed: float
    cutoff_speed: float


@dataclass(frozen=True)
class StateScales:
    """World-model normalisation (plan D3). Six entries each, for
    [root pos, root rotvec, root lin vel, root ang vel, joint pos, joint vel] (err) and the
    corresponding output / input groups; see ``world_model/state_spec.py``."""

    err_scale: tuple[float, float, float, float, float, float]
    out_scale: tuple[float, float, float, float, float, float]  # [dv_h, dw_b, dqd, dp_h, dth_b, dq]
    in_scale: tuple[float, float, float, float, float, float]  # [root_z, rot6d, lin_vel_h, ang_vel_b, q, qd]
    xi_friction: tuple[float, float]  # (offset, scale)
    xi_bias: tuple[float, float]
    xi_com: tuple[float, float]
    xi_delay: tuple[float, float] = (0.0, 1.0)  # action delay in physics substeps (offset, scale)


@dataclass(frozen=True)
class ObsNoise:
    """Uniform half-widths of the observation noise (paper Table A.1)."""

    joint_pos: float
    joint_vel: float
    ang_vel: float
    gravity: float


@dataclass(frozen=True)
class DomainRandomization:
    friction_range: tuple[float, float]
    encoder_bias: float  # +- rad
    com_offset: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]  # x, y, z ranges (m)
    action_delay_substeps: tuple[int, int] = (0, 0)  # (min, max) physics substeps; (0, 0) = off
    eval_friction: float = 1.0


@dataclass(frozen=True)
class ResetNoise:
    pose_range: dict[str, tuple[float, float]]
    velocity_range: dict[str, tuple[float, float]]
    joint_position_range: tuple[float, float]


@dataclass(frozen=True)
class SimSettings:
    timestep: float = 0.005
    decimation: int = 4
    nconmax: int = 35
    njmax: int = 250
    iterations: int = 10
    ls_iterations: int = 20
    ccd_iterations: int = 50
    contact_sensor_maxmatch: int = 64

    @property
    def control_dt(self) -> float:
        return self.timestep * self.decimation


@dataclass(frozen=True)
class RobotSpec:
    name: str
    xml_path: Path
    joint_names: tuple[str, ...]
    root_body: str
    anchor_body: str
    tracked_bodies: tuple[str, ...]
    foot_geom_regex: str
    com_body: str
    action_scale: float | dict[str, float]
    entity_cfg_fn: Callable[[bool], "EntityCfg"]  # arg: domain randomisation on (action delay)
    root_height_threshold: float
    root_tilt_threshold: float
    state_scales: StateScales
    obs_noise: ObsNoise
    dr: DomainRandomization
    reset_noise: ResetNoise
    # Reward stds / weights (dt-scaled BeyondMimic kernel; joint-space terms optional).
    reward_std_body_pos: float
    reward_std_body_ori: float
    reward_std_body_lin_vel: float
    reward_std_body_ang_vel: float
    reward_std_root_pos: float
    reward_std_root_ori: float
    reward_std_joint_pos: float = 1.0
    reward_std_joint_vel: float = 1.0
    reward_w_joint_pos: float = 0.0
    reward_w_joint_vel: float = 0.0
    reward_w_action_rate: float = -0.1
    reward_w_joint_limit: float = -2.0
    # Torque above the actuator's continuous rating (optional):
    # weight on sum_j relu(|tau_j| - torque_threshold)^2; 0 = off (G1). Needs ``torque_model``.
    reward_w_torque_headroom: float = 0.0
    torque_threshold: float = 0.0
    torque_model: TorqueModelSpec | None = None
    # Policy input scales: obs terms [q, qd, ang_vel, gravity, actions]; goal [q, qd, grav, ang_b, lin_h, height].
    obs_term_scale: tuple[float, float, float, float, float] = (1.0, 5.0, 2.0, 1.0, 1.0)
    goal_term_scale: tuple[float, float, float, float, float, float] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    # Support-foot termination (optional): bodies whose world z must follow the reference; () = off.
    support_feet: tuple[str, ...] = ()
    support_foot_threshold: float = 0.0
    sim: SimSettings = field(default_factory=SimSettings)
    stand_height: float = 0.0  # nominal root height when standing (m)
    viewer_distance: float = 2.8
    # Knock-down pushes for get-up training (plan D22), in m/s; sized per robot.
    push_speed: tuple[float, float] = (4.0, 6.0)
    push_ang_speed: float = 3.0
    push_down_speed: tuple[float, float] = (1.5, 3.0)
    lie_height: tuple[float, float] = (0.35, 0.55)
    # Mirror map for left-right data augmentation: joint permutation and signs, body permutation.
    mirror_joint_perm: tuple[int, ...] | None = None
    mirror_joint_sign: tuple[float, ...] | None = None
    mirror_body_map: dict[str, str] | None = None

    # ------------------------------------------------------------------ derived dimensions
    @property
    def nj(self) -> int:
        return len(self.joint_names)

    @property
    def nq(self) -> int:
        return 7 + self.nj

    @property
    def nv(self) -> int:
        return 6 + self.nj

    @property
    def state_dim(self) -> int:
        return self.nq + self.nv

    @property
    def delta_dim(self) -> int:
        return 12 + 2 * self.nj

    @property
    def feat_dim(self) -> int:
        return 13 + 2 * self.nj

    @property
    def obs_dim(self) -> int:
        return 3 * self.nj + 6

    @property
    def obs_terms(self) -> tuple[tuple[str, int], ...]:
        nj = self.nj
        return (("joint_pos", nj), ("joint_vel", nj), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", nj))

    @property
    def goal_feat_dim(self) -> int:
        return 2 * self.nj + 10

    @property
    def has_action_delay(self) -> bool:
        return self.dr.action_delay_substeps[1] > 0

    @property
    def xi_dim(self) -> int:
        return 1 + self.nj + 3 + (1 if self.has_action_delay else 0)

    @property
    def obs_noise_half(self) -> tuple[float, ...]:
        n = self.obs_noise
        return (n.joint_pos,) * self.nj + (n.joint_vel,) * self.nj + (n.ang_vel,) * 3 + (n.gravity,) * 3 + (0.0,) * self.nj

    def action_scale_dict(self) -> dict[str, float]:
        if isinstance(self.action_scale, dict):
            return dict(self.action_scale)
        return {j: float(self.action_scale) for j in self.joint_names}

    def load_model(self):
        """Standalone CPU ``mujoco.MjModel`` for FK / conversion; asserts the joint order."""
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "first joint must be the free joint"
        assert tuple(names[1:]) == self.joint_names, f"{self.name}: joint order mismatch: {names[1:]}"
        assert model.nq == self.nq and model.nv == self.nv, (model.nq, model.nv)
        return model
