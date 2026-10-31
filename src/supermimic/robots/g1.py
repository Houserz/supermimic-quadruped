"""Unitree G1 spec: the upstream SuperMimic reproduction's values, unchanged (regression baseline).

Every number here is what the upstream code hard-coded (plan §1.3): paper Table A.1–A.3
observation noise, domain randomisation and terminations, plan D3 world-model scales,
BeyondMimic / mjlab reward stds, and mjlab's G1 asset.
"""

from __future__ import annotations

from pathlib import Path

from supermimic.robots.spec import DomainRandomization, ObsNoise, ResetNoise, RobotSpec, SimSettings, StateScales

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def g1_xml_path() -> Path:
    import mjlab

    return Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"


def _g1_entity_cfg(domain_randomization: bool = True):
    from mjlab.asset_zoo.robots import get_g1_robot_cfg

    return get_g1_robot_cfg()


def _g1_action_scale() -> dict[str, float]:
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE

    return dict(G1_ACTION_SCALE)


G1_SPEC = RobotSpec(
    name="g1",
    xml_path=Path(__file__),  # resolved in g1_spec() (needs mjlab)
    joint_names=G1_JOINT_NAMES,
    root_body="pelvis",
    anchor_body="torso_link",
    tracked_bodies=G1_TRACKED_BODIES,
    foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
    com_body="torso_link",
    action_scale={},  # mjlab's G1_ACTION_SCALE (0.25 * effort / kp per joint group), filled in g1_spec()
    entity_cfg_fn=_g1_entity_cfg,
    root_height_threshold=0.25,  # paper App. A
    root_tilt_threshold=0.8,
    state_scales=StateScales(
        err_scale=(0.05, 0.1, 0.5, 1.0, 0.1, 2.0),
        out_scale=(0.3, 0.5, 2.0, 0.01, 0.02, 0.02),
        in_scale=(1.0, 1.0, 1.0, 2.0, 1.0, 5.0),
        xi_friction=(0.75, 0.45),
        xi_bias=(0.0, 0.01),
        xi_com=(0.0, 0.05),
    ),
    obs_noise=ObsNoise(joint_pos=0.01, joint_vel=0.5, ang_vel=0.2, gravity=0.05),  # Table A.1
    dr=DomainRandomization(
        friction_range=(0.3, 1.2),  # Table A.3
        encoder_bias=0.01,
        com_offset=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
        action_delay_substeps=(0, 0),
        eval_friction=1.0,
    ),
    reset_noise=ResetNoise(
        pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
        velocity_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)},
        joint_position_range=(-0.1, 0.1),
    ),
    reward_std_body_pos=0.3,  # Table A.2
    reward_std_body_ori=0.4,
    reward_std_body_lin_vel=1.0,
    reward_std_body_ang_vel=3.14,
    reward_std_root_pos=0.3,
    reward_std_root_ori=0.4,
    sim=SimSettings(timestep=0.005, decimation=4, nconmax=35, njmax=250, iterations=10, ls_iterations=20),
    stand_height=0.78,
    viewer_distance=2.8,
    push_speed=(4.0, 6.0),
    push_ang_speed=3.0,
    push_down_speed=(1.5, 3.0),
    lie_height=(0.35, 0.55),
)


def g1_spec() -> RobotSpec:
    """G1 spec with the xml path resolved (mjlab import deferred to first use)."""
    from dataclasses import replace

    return replace(G1_SPEC, xml_path=g1_xml_path(), action_scale=_g1_action_scale())
