"""Dobot Rover X1 (12-DoF quadruped) spec.

The MJCF under ``assets/rover/`` is the model the motion library is retargeted on
(``configs/robots/rover.yaml``; provenance in ``assets/rover/SOURCE.md``). It has no actuators, so
the PD actuators below are the only place gains and torque limits come from. Every tuning number
says where it comes from:

* **[dobot]**     Dobot's official data in ``Dobot-Team/dobot_rover_simulation``: the URDF
  (``dobot_rl_gym/resources/robots/dobot/urdf/dobot_quad_ros.urdf``) or the locomotion training
  config (``dobot_rl_gym/legged_gym/envs/dobot/dobot_config.py``);
* **[walk]**      the PD gains of a locally trained RL walking policy;
* **[trackerlab Go2]**  ``Renforce-Dynamics/trackerlab-supermimic`` ``docs/go2.md``, the one
  public SuperMimic tracking run on a quadruped (Go2 trot, return 0.927);
* **[mjlab go1]** ``mjlab/asset_zoo/robots/unitree_go1/go1_constants.py``;
* **[scaled]**    a G1 / paper value scaled by the standing height (0.391 m vs the G1's 0.78 m:
  lengths x 0.5, velocities x sqrt(0.5) ~ 0.7 under dynamic similarity);
* **[measured]**  read off the library or the simulator by ``scripts/calibrate_scales.py``.
"""

from __future__ import annotations

from pathlib import Path

from supermimic.robots.spec import DomainRandomization, ObsNoise, ResetNoise, RobotSpec, SimSettings, StateScales

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "rover"
ROVER_XML = ASSET_DIR / "dobot.xml"

LEGS: tuple[str, ...] = ("FL", "FR", "RL", "RR")
_WORD = {"FL": "front_left", "FR": "front_right", "RL": "rear_left", "RR": "rear_right"}
JOINT_NAMES: tuple[str, ...] = tuple(f"joint_{_WORD[l]}_{j}" for l in LEGS for j in ("abad", "thigh_pitch", "calf_pitch"))

# [dobot] default_joint_angles of Dobot's locomotion config: knees backwards on every leg, the
# branch the motion library was retargeted on (roverdata.robot KNEE_SIGN).
HOME: dict[str, float] = {}
for _l in LEGS:
    _front = _l[0] == "F"
    HOME[f"joint_{_WORD[_l]}_abad"] = 0.0
    HOME[f"joint_{_WORD[_l]}_thigh_pitch"] = 0.84 if _front else 0.75
    HOME[f"joint_{_WORD[_l]}_calf_pitch"] = -1.3 if _front else -1.22
STAND_Z = 0.391  # root height of the library's nominal stance (thigh 0.84, calf -1.3, foot radius 0.028)

ROOT_BODY = "link_trunk"
# The feet are spheres on the calf bodies, so there are no foot bodies to track; the calf's
# position (the knee) and orientation together fix the foot, as the G1's ankle links do.
TRACKED_BODIES: tuple[str, ...] = (ROOT_BODY,) + tuple(f"link_{_WORD[l]}_{p}" for p in ("thigh", "calf") for l in LEGS)
FOOT_GEOM_REGEX = r"^(FL|FR|RL|RR)_foot_collision$"
_COLLISION_REGEX = r".*_collision"

STIFFNESS = 25.0  # [walk] every joint
DAMPING = 1.3  # [walk]
EFFORT_HIP = 23.0  # [dobot] URDF <limit effort> abad and thigh
EFFORT_KNEE = 55.0  # [dobot] URDF <limit effort> calf
ACTION_SCALE = 0.25  # [dobot] action_scale


def get_spec():
    import mujoco

    return mujoco.MjSpec.from_file(str(ROVER_XML))


def _collision_cfg():
    """[mjlab go1] FULL_COLLISION: every *_collision geom collides (self-collision on), feet with
    condim 6 and friction (1, 5e-3, 5e-4), everything else frictionless."""
    from mjlab.utils.spec_config import CollisionCfg

    return CollisionCfg(
        geom_names_expr=(_COLLISION_REGEX,),
        contype=1,
        conaffinity=1,
        solref=(0.01, 1),
        condim={FOOT_GEOM_REGEX: 6, _COLLISION_REGEX: 1},
        priority={FOOT_GEOM_REGEX: 1, ".*": 0},
        friction={FOOT_GEOM_REGEX: (1, 5e-3, 5e-4)},
    )


def damping_for(stiffness: float) -> float:
    """kd that keeps the [walk] damping ratio at another kp (kd proportional to sqrt(kp))."""
    return DAMPING * (stiffness / STIFFNESS) ** 0.5


def get_rover_entity_cfg(domain_randomization: bool = True, stiffness: float = STIFFNESS):
    """Fresh EntityCfg: MuJoCo position actuators, kp 25 / kd 1.3, torque-limited per joint group.

    Armature and joint friction stay as in the XML (none; passive damping 0.02): no identified
    values are used.
    """
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    del domain_randomization  # no action delay on this robot yet (spec.dr.action_delay_substeps)
    kd = damping_for(stiffness)
    hip = BuiltinPositionActuatorCfg(target_names_expr=(r".*_abad", r".*_thigh_pitch"), stiffness=stiffness, damping=kd, effort_limit=EFFORT_HIP)
    knee = BuiltinPositionActuatorCfg(target_names_expr=(r".*_calf_pitch",), stiffness=stiffness, damping=kd, effort_limit=EFFORT_KNEE)
    # Soft limit factor 1.0: the clips are cut to the model's hard range and a
    # 0.9 soft range would penalise tracking them.
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, STAND_Z + 0.01), joint_pos=dict(HOME), joint_vel={".*": 0.0}),
        collisions=(_collision_cfg(),),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=(hip, knee), soft_joint_pos_limit_factor=1.0),
    )


# ---------------------------------------------------------------- mirror map
# Mirror plane x-z: legs swap sides; the abad hinge (axis x) negates, the pitch hinges (axis y)
# keep their sign (pseudovector rule). Same map as roverdata.robot, which checks it by FK;
# tests/test_rover_spec.py checks it again on this copy of the model.
_PAIR = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
MIRROR_JOINT_PERM: tuple[int, ...] = tuple(3 * LEGS.index(_PAIR[l]) + k for l in LEGS for k in range(3))
MIRROR_JOINT_SIGN: tuple[float, ...] = (-1.0, 1.0, 1.0) * 4


def _mirror_body_map() -> dict[str, str]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(ROVER_XML))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    swap = {"_left_": "_right_", "_right_": "_left_"}
    out = {}
    for n in names:
        partner = n
        for a, b in swap.items():
            if a in n:
                partner = n.replace(a, b)
                break
        assert partner in names, (n, partner)
        out[n] = partner
    return out


ROVER_SPEC = RobotSpec(
    name="rover",
    xml_path=ROVER_XML,
    joint_names=JOINT_NAMES,
    root_body=ROOT_BODY,
    anchor_body=ROOT_BODY,
    tracked_bodies=TRACKED_BODIES,
    foot_geom_regex=FOOT_GEOM_REGEX,
    com_body=ROOT_BODY,
    action_scale=ACTION_SCALE,
    entity_cfg_fn=get_rover_entity_cfg,
    # Terminations. Height [scaled] G1 0.25 m x 0.5 = 0.125, raised to 0.15 toward [trackerlab Go2]
    # 0.16 m. Tilt: 1 - cos(60 deg) = 0.5 (G1 0.8 ~ 78 deg is beyond where a quadruped recovers).
    root_height_threshold=0.15,
    root_tilt_threshold=0.5,
    # World-model normalisation (plan D3). MEASURED by scripts/calibrate_scales.py (one H200,
    # 2026-10-29: 256 envs x 300 steps, all 566 library clips, kp 25), p95 of the
    # per-step quantities as reference replay / random policy:
    #   per-step change   root pos 0.014/0.027 m, rotvec 0.032/0.029 rad, lin vel 0.098/0.104 m/s,
    #                     ang vel 0.55/1.58 rad/s, q 0.071/0.065 rad, qd 2.2/3.7 rad/s
    #   integration resid dp 0.00073/0.00074 m, dth 0.0040/0.0078 rad, dq 0.017/0.024 rad
    #   state rms         root_z 0.355/0.339 m, lin vel 0.38/0.51 m/s, ang vel 0.76/0.72 rad/s, q 0.88/0.96, qd 1.6/1.7
    # Each scale sits at about the p95 of its quantity, between the two regimes where they differ;
    # the inputs at about their rms.
    state_scales=StateScales(
        err_scale=(0.02, 0.05, 0.1, 1.0, 0.1, 3.0),
        out_scale=(0.1, 1.0, 3.0, 0.001, 0.005, 0.02),
        in_scale=(0.35, 1.0, 0.5, 1.0, 1.0, 2.5),
        xi_friction=(0.75, 0.45),  # centre / half-width of the friction range below
        xi_bias=(0.0, 0.01),
        xi_com=(0.0, 0.03),
    ),
    obs_noise=ObsNoise(joint_pos=0.01, joint_vel=0.5, ang_vel=0.2, gravity=0.05),  # paper Table A.1
    dr=DomainRandomization(
        friction_range=(0.3, 1.2),  # paper Table A.3
        encoder_bias=0.01,  # paper
        # [scaled] G1 +-0.025/0.05/0.05 m on a ~0.4 m torso; the Rover's trunk box is 0.70 x 0.18 x 0.12 m.
        com_offset=((-0.03, 0.03), (-0.02, 0.02), (-0.02, 0.02)),
        action_delay_substeps=(0, 0),  # off until a latency is measured on the robot
        eval_friction=1.0,
    ),
    reset_noise=ResetNoise(  # [scaled] G1 ranges, positions x 0.5, velocities x 0.7
        pose_range={"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.005, 0.005), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
        velocity_range={"x": (-0.35, 0.35), "y": (-0.35, 0.35), "z": (-0.15, 0.15), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)},
        joint_position_range=(-0.1, 0.1),
    ),
    # Reward kernel stds: [trackerlab Go2] position 0.15 m, linear velocity 0.5 m/s, root position
    # 0.2 m (G1 0.3 / 1.0 / 0.3 [scaled] give 0.15 / 0.7 / 0.15); orientations and angular velocity paper.
    # Not re-derived from the calibration's replay errors (body pos rms 0.10 m, root pos 0.35 m): at
    # kp 25 the open-loop replay sags and falls (see rover_spec), so those errors measure the fall.
    reward_std_body_pos=0.15,
    reward_std_body_ori=0.4,
    reward_std_body_lin_vel=0.5,
    reward_std_body_ang_vel=3.14,
    reward_std_root_pos=0.2,
    reward_std_root_ori=0.4,
    reward_w_action_rate=-0.1,
    reward_w_joint_limit=-2.0,
    obs_term_scale=(1.0, 5.0, 2.0, 1.0, 1.0),
    goal_term_scale=(1.0, 5.0, 1.0, 2.0, 1.0, 2.5),  # root height ~0.4 m x 2.5 -> O(1)
    sim=SimSettings(timestep=0.005, decimation=4, nconmax=35, njmax=250, iterations=10, ls_iterations=20),  # [dobot] dt 0.005, decimation 4
    stand_height=STAND_Z,
    viewer_distance=1.6,
    # Get-up pushes (plan D22) [scaled].
    push_speed=(2.8, 4.2),
    push_ang_speed=3.0,
    push_down_speed=(1.0, 2.1),
    lie_height=(0.08, 0.2),
    mirror_joint_perm=MIRROR_JOINT_PERM,
    mirror_joint_sign=MIRROR_JOINT_SIGN,
    mirror_body_map=None,  # filled by rover_spec() (needs the compiled model)
)


def rover_spec(stiffness: float | None = None) -> RobotSpec:
    """The Rover spec, optionally at another PD stiffness (``robots.set_robot_overrides``; the
    damping follows ``damping_for``). The open-loop feasibility screen needs one: at the spec's
    kp 25 a PD replay of a standing clip sags and folds a rear leg within 0.7 s (root
    0.425 -> 0.296 m, rear-left calf error 0.72 rad, ai4a_D1_001_KAN01_001_s0), so it measures
    the softness of the servo rather than the clip. A policy supplies that holding offset."""
    from dataclasses import replace
    from functools import partial

    spec = replace(ROVER_SPEC, mirror_body_map=_mirror_body_map())
    if stiffness is not None:
        spec = replace(spec, entity_cfg_fn=partial(get_rover_entity_cfg, stiffness=float(stiffness)))
    return spec
