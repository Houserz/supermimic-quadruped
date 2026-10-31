"""Phase 1.5: measure the typical magnitudes the world-model / policy scales must normalise.

Collects states in the robot's env under two regimes -- the reference played back open-loop
(action = next reference pose) and a random policy -- and prints, per state group, the RMS of
the quantity itself (-> ``in_scale``), of its per-step change (-> ``out_scale`` for the velocity
groups, ``err_scale`` for the one-step prediction error), and of the tracked-body errors seen by
the reward kernel. Also reports the per-step *integration residual*: the position change not
explained by ``dt * v`` (-> ``out_scale`` for the dp / dth / dq groups).

  uv run python scripts/calibrate_scales.py --robot rover --npz-dir data/npz/rover --num-envs 256 --steps 300

The numbers go into ``src/supermimic/robots/<robot>.py`` with a comment quoting this run.
"""

from __future__ import annotations

from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.state_io import StateIO
from supermimic.kinematics.rotations import quat_apply_inverse, quat_mul, rotvec_from_quat
from supermimic.robots import get_robot_spec
from supermimic.world_model.state_spec import quat_apply_inverse_quat


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt((x.double() ** 2).mean()))


def _p95(x: torch.Tensor) -> float:
    return float(x.abs().flatten().float().quantile(0.95))


def main(robot: str = "rover", npz_dir: Path = Path("data/npz/rover"), num_envs: int = 256, steps: int = 300, device: str = "cuda:0", manifest: Path | None = None) -> None:
    spec = get_robot_spec(robot)
    nj = spec.nj
    files = [str(f) for f in discover_npz(npz_dir, manifest=manifest)]
    env = make_env(SuperMimicEnvCfgArgs(npz_files=files, robot=robot, num_envs=num_envs, obs_noise=False, action_noise_std=0.0, record_transitions=False, episode_length_s=steps * spec.sim.control_dt + 1, seed=0), device=device)
    io = StateIO(env)
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    q0 = env.scene["robot"].data.default_joint_pos[0]
    scale = torch.tensor([spec.action_scale_dict()[j] for j in spec.joint_names], device=device)
    dt = spec.sim.control_dt
    G = lib.goal_feat_dim

    def ref_action(obs):
        g = obs["goal"][:, G : G + nj]  # q_ref(t+1) - q0 of the next target frame
        return g / scale

    def rand_action(obs):
        return 0.5 * torch.randn(num_envs, nj, device=device)

    for regime, act_fn in (("reference-replay", ref_action), ("random-policy", rand_action)):
        obs, _ = env.reset()
        S, dS, resid, errs = [], [], [], {"body_pos": [], "body_lin": [], "body_ang": [], "root_pos": [], "joint_pos": [], "joint_vel": []}
        prev = io.read()
        for _ in range(steps):
            obs, rew, term, tout, _ = env.step(act_fn(obs))
            s = io.read()
            alive = ~env.reset_buf
            S.append(s[alive])
            p, q, qj, v, w, qd = (prev[alive][..., sl] for sl in (slice(0, 3), slice(3, 7), slice(7, 7 + nj), slice(7 + nj, 10 + nj), slice(10 + nj, 13 + nj), slice(13 + nj, 13 + 2 * nj)))
            p2, q2, qj2, v2, w2, qd2 = (s[alive][..., sl] for sl in (slice(0, 3), slice(3, 7), slice(7, 7 + nj), slice(7 + nj, 10 + nj), slice(10 + nj, 13 + nj), slice(13 + nj, 13 + 2 * nj)))
            rot = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q), q2))
            dS.append(torch.cat([p2 - p, rot, v2 - v, w2 - w, qj2 - qj, qd2 - qd], -1))
            # integration residual: what explicit integration with the new velocity leaves over
            resid.append(torch.cat([p2 - p - dt * v2, rot - dt * w2, qj2 - qj - dt * qd2], -1))
            errs["root_pos"].append((cmd.root_pos_w - cmd.robot_root_pos_w)[alive].norm(dim=-1))
            errs["body_pos"].append((cmd.body_pos_relative_w - cmd.robot_body_pos_w)[alive].norm(dim=-1))
            errs["body_lin"].append((cmd.body_lin_vel_w - cmd.robot_body_lin_vel_w)[alive].norm(dim=-1))
            errs["body_ang"].append((cmd.body_ang_vel_w - cmd.robot_body_ang_vel_w)[alive].norm(dim=-1))
            errs["joint_pos"].append((cmd.joint_pos - cmd.robot_joint_pos)[alive])
            errs["joint_vel"].append((cmd.joint_vel - cmd.robot_joint_vel)[alive])
            prev = s
        S = torch.cat(S)
        dS = torch.cat(dS)
        resid = torch.cat(resid)
        groups = {"root_pos": (0, 3), "root_rotvec": (3, 6), "root_lin_vel": (6, 9), "root_ang_vel": (9, 12), "joint_pos": (12, 12 + nj), "joint_vel": (12 + nj, 12 + 2 * nj)}
        print(f"\n=== {regime}: {S.shape[0]} samples ===")
        print("state magnitudes (rms / p95):")
        for name, (a, b) in {"root_z": (2, 3), "root_lin_vel_w": (7 + nj, 10 + nj), "root_ang_vel_b": (10 + nj, 13 + nj), "joint_pos": (7, 7 + nj), "joint_vel": (13 + nj, 13 + 2 * nj)}.items():
            print(f"  {name:16s} rms {_rms(S[:, a:b]):9.4f}  p95 {_p95(S[:, a:b]):9.4f}")
        print("per-step change (rms / p95)  -> err_scale candidates, out_scale for the velocity groups:")
        for name, (a, b) in groups.items():
            print(f"  {name:16s} rms {_rms(dS[:, a:b]):9.5f}  p95 {_p95(dS[:, a:b]):9.5f}")
        print("integration residual (rms / p95)  -> out_scale for dp_h / dth_b / dq:")
        for name, (a, b) in {"dp": (0, 3), "dth": (3, 6), "dq": (6, 6 + nj)}.items():
            print(f"  {name:16s} rms {_rms(resid[:, a:b]):9.6f}  p95 {_p95(resid[:, a:b]):9.6f}")
        print("tracking errors seen by the reward (rms / p95)  -> reward stds:")
        for k, v in errs.items():
            v = torch.cat(v)
            print(f"  {k:16s} rms {_rms(v):9.4f}  p95 {_p95(v):9.4f}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
