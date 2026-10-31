"""SuperMimic tracking environment on mjlab (plan Phase 2.3), robot-generic (plan decision B).

Differences from mjlab's BeyondMimic re-implementation (``mjlab.tasks.tracking``):
  * observations: paper Table A.1 proprioception only (no anchor pose, no base lin vel),
    10-frame history (term-major), plus a separate ``goal`` group with K future frames;
  * action: ``NoisyJointPositionAction`` (clean action observed, noise only executed);
  * multi-clip command with return-based reset sampling and clip-end time-outs;
  * rewards: self-collision term removed, joint-limit weight -2; optional joint-space terms;
  * terminations: root height / tilt vs reference, optional support-foot term, clip-end
    time-out; optionally disabled for a random fraction of episodes (get-up training, plan D22);
  * events: no push by default (optional large knock-down pushes for get-up training, plan D22);
    friction and encoder bias re-sampled per episode; COM at startup; optional action delay;
  * a ``ReplayRecorder`` streams transitions into a GPU replay buffer.

Every robot-dependent name, threshold, range and scale comes from ``robots.get_robot_spec``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.envs.mdp import events as sm_events
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp import rewards as sm_rewards
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg
from supermimic.robots import RobotSpec, get_robot_spec


class QuietSimulationCfg(SimulationCfg):
    """SimulationCfg that disables MuJoCo Warp's per-step overflow printf (a kernel build-time constant,
    so it must be set before the CUDA graphs are captured)."""

    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = 0


# Upstream G1 constants, kept for the existing tests / callers.
from supermimic.robots.g1 import G1_TRACKED_BODIES as TRACKED_BODIES  # noqa: E402

ANCHOR_BODY = "torso_link"
ROOT_BODY = "pelvis"
FOOT_GEOM_REGEX = r"^(left|right)_foot[1-7]_collision$"


@dataclass
class SuperMimicEnvCfgArgs:
    npz_files: list[str]
    robot: str = "g1"
    num_envs: int = 4096
    future_frames: int = 10
    history_length: int = 10
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"  # return | uniform | start
    replay_capacity: int = 1000
    record_transitions: bool = True
    seed: int | None = 0
    root_height_threshold: float | None = None  # None = robot spec
    root_tilt_threshold: float | None = None
    support_foot_threshold: float | None = None  # None = robot spec; 0 disables
    joint_limit_weight: float | None = None
    bin_frames: int = 50
    compact_library: bool = False
    anchor_obs: bool = False  # add BeyondMimic anchor-relative pose (9) to the goal group (plan D5 alt.)
    immune_probability: float = 0.0  # episodes whose root terminations are disabled (get-up, plan D22)
    lie_probability: float = 0.0  # fraction of resets that start lying on the floor, always immune (D22)
    push_interval_s: tuple[float, float] | None = None  # knock-down pushes per env (D22); None = off
    push_speed: tuple[float, float] | None = None  # horizontal kick (m/s); None = robot spec
    push_ang_speed: float | None = None  # roll / pitch kick half-range (rad/s)
    push_down_prob: float = 0.3  # fraction of pushes that also kick downwards
    push_down_speed: tuple[float, float] | None = None  # downward kick (m/s)
    clip_weight_alpha: float = 1.0  # plan 2.6: p(clip) ∝ duration^alpha (1 = uniform over frames)
    static_bin_factor: float = 1.0  # plan 2.6: sampling weight multiplier of static bins
    clip_weights: dict | None = None  # optional per-clip multipliers (manifest ``weight`` column)
    debug_vis: bool = False  # draw the reference ghost in the viewer (scripts/play.py)
    extra_command_kwargs: dict = field(default_factory=dict)


def make_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    spec: RobotSpec = get_robot_spec(a.robot)
    nz = spec.obs_noise
    height_thr = spec.root_height_threshold if a.root_height_threshold is None else a.root_height_threshold
    tilt_thr = spec.root_tilt_threshold if a.root_tilt_threshold is None else a.root_tilt_threshold
    foot_thr = spec.support_foot_threshold if a.support_foot_threshold is None else a.support_foot_threshold
    use_feet = bool(spec.support_feet) and foot_thr > 0
    joint_limit_w = spec.reward_w_joint_limit if a.joint_limit_weight is None else a.joint_limit_weight
    reward_cfg = sm_rewards.TrackingRewardCfg.from_robot(spec, dt=spec.sim.control_dt)

    # ---------------------------------------------------------------- observations
    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)
    policy_terms = {
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-nz.joint_pos, nz.joint_pos)),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, noise=noise(-nz.joint_vel, nz.joint_vel)),
        "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel, noise=noise(-nz.ang_vel, nz.ang_vel)),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity, noise=noise(-nz.gravity, nz.gravity)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
            history_length=a.history_length,
            flatten_history_dim=True,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "future_frames": ObservationTermCfg(func=sm_obs.goal_future_frames, params={"command_name": "motion"}),
                **(
                    {
                        "anchor_pos": ObservationTermCfg(func=track_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=noise(-0.25, 0.25)),
                        "anchor_ori": ObservationTermCfg(func=track_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=noise(-0.05, 0.05)),
                    }
                    if a.anchor_obs
                    else {}
                ),
            },
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
        ),
    }

    # ---------------------------------------------------------------- actions
    actions = {
        "joint_pos": NoisyJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=spec.action_scale_dict(),
            use_default_offset=True,
            noise_std=a.action_noise_std,
        )
    }

    # ---------------------------------------------------------------- commands
    rn = spec.reset_noise
    commands = {
        "motion": MultiClipMotionCommandCfg(
            npz_files=[str(f) for f in a.npz_files],
            anchor_body_name=spec.anchor_body,
            root_body_name=spec.root_body,
            body_names=spec.tracked_bodies,
            entity_name="robot",
            robot=spec.name,
            future_frames=a.future_frames,
            bin_frames=a.bin_frames,
            pose_range=dict(rn.pose_range) if a.reset_noise else {},
            velocity_range=dict(rn.velocity_range) if a.reset_noise else {},
            joint_position_range=rn.joint_position_range if a.reset_noise else (0.0, 0.0),
            sampling_mode=a.sampling_mode,  # type: ignore[arg-type]
            immune_probability=a.immune_probability,
            lie_probability=a.lie_probability,
            lie_height=spec.lie_height,
            compact_library=a.compact_library,
            clip_weight_alpha=a.clip_weight_alpha,
            static_bin_factor=a.static_bin_factor,
            clip_weights=a.clip_weights,
            debug_vis=a.debug_vis,
            **a.extra_command_kwargs,
        )
    }

    # ---------------------------------------------------------------- events (plan D8)
    d = spec.dr
    events: dict[str, EventTermCfg] = {}
    if a.domain_randomization:
        events["base_com"] = EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(spec.com_body,)),
                "operation": "add",
                "ranges": {0: d.com_offset[0], 1: d.com_offset[1], 2: d.com_offset[2]},
            },
        )
        events["encoder_bias"] = EventTermCfg(
            mode="reset",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-d.encoder_bias, d.encoder_bias)},
        )
        events["foot_friction"] = EventTermCfg(
            mode="reset",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=spec.foot_geom_regex),
                "operation": "abs",
                "ranges": d.friction_range,
                "shared_random": True,
            },
        )
        if d.action_delay_substeps[1] > 0:
            # One command lag per episode (xi entry); the actuator cfg holds it (delay_hold_prob=1.0).
            events["action_delay"] = EventTermCfg(mode="reset", func=sm_events.action_delay, params={"lag_range": d.action_delay_substeps})
    else:
        # Evaluation protocol: foot friction fixed, no other randomization.
        events["foot_friction"] = EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=spec.foot_geom_regex),
                "operation": "abs",
                "ranges": (d.eval_friction, d.eval_friction),
                "shared_random": True,
            },
        )

    if a.push_interval_s is not None:
        events["push"] = EventTermCfg(
            mode="interval",
            func=sm_events.big_push,
            interval_range_s=a.push_interval_s,
            params={
                "speed_range": a.push_speed or spec.push_speed,
                "ang_speed": spec.push_ang_speed if a.push_ang_speed is None else a.push_ang_speed,
                "down_prob": a.push_down_prob,
                "down_speed_range": a.push_down_speed or spec.push_down_speed,
            },
        )

    # ---------------------------------------------------------------- rewards (Table A.2 + optional joint terms)
    rc = reward_cfg
    rewards = {
        "motion_global_root_pos": RewardTermCfg(func=track_mdp.motion_global_anchor_position_error_exp, weight=rc.w_root_pos, params={"command_name": "motion", "std": rc.std_root_pos}),
        "motion_global_root_ori": RewardTermCfg(func=track_mdp.motion_global_anchor_orientation_error_exp, weight=rc.w_root_ori, params={"command_name": "motion", "std": rc.std_root_ori}),
        "motion_body_pos": RewardTermCfg(func=track_mdp.motion_relative_body_position_error_exp, weight=rc.w_body_pos, params={"command_name": "motion", "std": rc.std_body_pos}),
        "motion_body_ori": RewardTermCfg(func=track_mdp.motion_relative_body_orientation_error_exp, weight=rc.w_body_ori, params={"command_name": "motion", "std": rc.std_body_ori}),
        "motion_body_lin_vel": RewardTermCfg(func=track_mdp.motion_global_body_linear_velocity_error_exp, weight=rc.w_body_lin_vel, params={"command_name": "motion", "std": rc.std_body_lin_vel}),
        "motion_body_ang_vel": RewardTermCfg(func=track_mdp.motion_global_body_angular_velocity_error_exp, weight=rc.w_body_ang_vel, params={"command_name": "motion", "std": rc.std_body_ang_vel}),
    }
    if rc.w_joint_pos != 0.0:
        rewards["motion_joint_pos"] = RewardTermCfg(func=sm_rewards.motion_joint_pos_error_exp, weight=rc.w_joint_pos, params={"command_name": "motion", "std": rc.std_joint_pos})
    if rc.w_joint_vel != 0.0:
        rewards["motion_joint_vel"] = RewardTermCfg(func=sm_rewards.motion_joint_vel_error_exp, weight=rc.w_joint_vel, params={"command_name": "motion", "std": rc.std_joint_vel})
    rewards["action_rate_l2"] = RewardTermCfg(func=base_mdp.action_rate_l2, weight=rc.w_action_rate)
    rewards["joint_limit"] = RewardTermCfg(func=base_mdp.joint_pos_limits, weight=joint_limit_w, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))})
    if rc.w_torque_headroom != 0.0:
        # optional: torque above the actuator's continuous rating; the analytic twin lives in heads.reward.
        rewards["torque_headroom"] = RewardTermCfg(func=sm_rewards.torque_headroom_cost, weight=rc.w_torque_headroom, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": rc.torque_threshold, "robot": a.robot})

    # ---------------------------------------------------------------- terminations
    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        # A world whose physics produced NaN/Inf is reset like a time-out (no failure label, no
        # sampler update); the replay buffer drops its non-finite rows (plan 3.3 run j0 went NaN).
        "nan": TerminationTermCfg(func=base_mdp.nan_detection, time_out=True),
        "clip_end": TerminationTermCfg(func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True),
        "root_height": TerminationTermCfg(func=sm_term.bad_root_height, params={"command_name": "motion", "threshold": height_thr}),
        "root_tilt": TerminationTermCfg(func=sm_term.bad_root_tilt, params={"command_name": "motion", "threshold": tilt_thr}),
    }
    if use_feet:
        terminations["support_foot"] = TerminationTermCfg(
            func=sm_term.bad_support_foot_height, params={"command_name": "motion", "threshold": foot_thr, "body_names": spec.support_feet}
        )

    recorders = {}
    if a.record_transitions:
        recorders["replay"] = ReplayRecorderCfg(
            params={
                "capacity": a.replay_capacity,
                "foot_geom_regex": spec.foot_geom_regex,
                "com_body_name": spec.com_body,
                "action_delay": spec.has_action_delay and a.domain_randomization,
            }
        )

    sim = spec.sim
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=a.num_envs, entities={"robot": spec.entity_cfg_fn(a.domain_randomization)}),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        recorders=recorders,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name=spec.anchor_body, distance=spec.viewer_distance, fovy=55.0, elevation=-5.0, azimuth=120.0),
        sim=QuietSimulationCfg(
            nconmax=sim.nconmax,
            njmax=sim.njmax,
            contact_sensor_maxmatch=sim.contact_sensor_maxmatch,
            mujoco=MujocoCfg(timestep=sim.timestep, iterations=sim.iterations, ls_iterations=sim.ls_iterations, ccd_iterations=sim.ccd_iterations),
        ),
        decimation=sim.decimation,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )
    return cfg


def make_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    return ManagerBasedRlEnv(make_env_cfg(a), device=device)


def make_heads(env: ManagerBasedRlEnv, robot: str, device: str | None = None, reward_dt: float | None = None):
    """AnalyticHeads for ``env``'s robot: FK from the spec's standalone MJCF, limits / q0 from the entity."""
    import torch

    from supermimic.kinematics.robot_fk import RobotKinematics
    from supermimic.world_model.heads import AnalyticHeads

    spec = get_robot_spec(robot)
    device = device or env.device
    entity = env.scene["robot"]
    fk = RobotKinematics(spec.load_model(), device=device)
    foot_thr = spec.support_foot_threshold if spec.support_feet else 0.0
    active = env.termination_manager.active_terms if hasattr(env, "termination_manager") else ()
    use_feet = "support_foot" in active
    return AnalyticHeads(
        fk,
        spec.tracked_bodies,
        spec.anchor_body,
        spec.root_body,
        soft_limits=entity.data.soft_joint_pos_limits[0].clone(),
        default_joint_pos=entity.data.default_joint_pos[0].clone(),
        reward_cfg=sm_rewards.TrackingRewardCfg.from_robot(spec, dt=reward_dt or spec.sim.control_dt),
        support_feet=spec.support_feet if use_feet else (),
        support_foot_threshold=env.termination_manager.get_term_cfg("support_foot").params["threshold"] if use_feet else 0.0,
        height_threshold=env.termination_manager.get_term_cfg("root_height").params["threshold"],
        tilt_threshold=env.termination_manager.get_term_cfg("root_tilt").params["threshold"],
        obs_noise_half=torch.tensor(spec.obs_noise_half, device=device),
        torque_model=sm_rewards.torque_model_for(spec, entity, device),
    )
