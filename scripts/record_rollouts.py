"""Record what a checkpoint actually does on each clip, for viewing next to the reference.

Same protocol as ``eval_benchmark.py`` (paper Appendix F): every clip played once from its first
frame with the deterministic policy, no noise, no randomisation, the checkpoint's own terminations.
Per control step the robot's root pose and joint angles are stored together with the frame of the
clip the command was at, until the clip ends or a training termination fires. The step on which an
env terminates is not stored: mjlab resets the env inside that step, so its state is already the
next episode's start.

  uv run python scripts/record_rollouts.py --ckpt logs/rover/ckpt_latest.pt --npz-dir data/npz/rover \
      --manifest data/npz/rover/manifest.csv --exclude '_mirror$'

Writes ``<ckpt dir>/rollouts_<ckpt stem>.npz``: ``names``, per clip ``start`` / ``count`` into the
sample arrays, ``clip_frames``, ``failed``, ``fail_step``; per sample ``ref_frame`` (the command's
frame of the clip), ``root_pos`` (env-local, m), ``root_quat`` (wxyz), ``joint_pos`` (spec joint order).

Sample k is the robot after k control steps, at the time of reference frame k; ``ref_frame`` reads
k + 1 because the command is the *target*: the reset writes frame 0 into the robot and the command
then steps to frame 1 (measured on the first Rover recording, job 400689: 1, 2, 3, ...).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots import get_robot_spec


@torch.no_grad()
def main(ckpt: Path, npz_dir: Path, manifest: Path | None = None, group: str | None = None, exclude: str | None = None, max_envs: int = 1024, device: str = "cuda:0", out: Path | None = None) -> None:
    files = [str(f) for f in discover_npz(npz_dir, manifest=manifest, group=group, exclude=exclude)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    robot_name = cfg.get("robot", "g1")
    spec = get_robot_spec(robot_name)
    e = cfg["env"]
    n_env = min(len(files), max_envs)
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=files, robot=robot_name, num_envs=n_env, future_frames=e["future_frames"], episode_length_s=1.0e5,
        action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False, sampling_mode="start",
        record_transitions=False, seed=0, anchor_obs=e.get("anchor_obs", False), support_foot_threshold=e.get("support_foot_threshold"),
        root_height_threshold=e.get("root_height_threshold"), root_tilt_threshold=e.get("root_tilt_threshold"),
    ), device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        robot = env.scene["robot"]
        assert tuple(robot.joint_names) == tuple(spec.joint_names), (robot.joint_names, spec.joint_names)
        origins = env.scene.env_origins
        names = [c.name for c in lib.clips]
        n_clips = len(names)
        per_clip: dict[int, dict] = {}
        for b0 in range(0, n_clips, n_env):
            ids = list(range(b0, min(b0 + n_env, n_clips)))
            pad = torch.tensor(ids + [ids[-1]] * (n_env - len(ids)), device=device)
            start = lib.clip_start[pad]
            length = lib.clip_len[pad]
            cmd.forced_frame[:] = start
            obs, _ = env.reset()
            horizon = int(length.max())
            buf = torch.zeros(horizon + 1, n_env, 3 + 4 + spec.nj + 1, device=device)
            alive = torch.zeros(horizon + 1, n_env, dtype=torch.bool, device=device)
            done = torch.zeros(n_env, dtype=torch.bool, device=device)
            failed = torch.zeros(n_env, dtype=torch.bool, device=device)
            fail_step = torch.full((n_env,), -1, dtype=torch.long, device=device)

            def snap(t: int, keep: torch.Tensor) -> None:
                buf[t, :, 0:3] = cmd.robot_root_pos_w - origins
                buf[t, :, 3:7] = cmd.robot_root_quat_w
                buf[t, :, 7 : 7 + spec.nj] = robot.data.joint_pos
                buf[t, :, -1] = (cmd.frame - start).float()
                alive[t] = keep

            snap(0, ~done)
            for t in range(1, horizon + 1):
                obs, _, term, tout, _ = env.step(policy(obs["policy"], obs["goal"]))
                ended = term | tout
                newly = ~done & term
                failed |= newly
                fail_step[newly] = t
                snap(t, ~done & ~ended)
                done |= ended
                if done.all():
                    break
            b, a = buf.cpu().numpy(), alive.cpu().numpy()
            for k, ci in enumerate(ids):
                rows = b[a[:, k], k]
                per_clip[ci] = {"rows": rows, "failed": bool(failed[k]), "fail_step": int(fail_step[k]), "frames": int(length[k])}
            print(f"  recorded {min(b0 + n_env, n_clips)}/{n_clips} clips")
        rows = [per_clip[i]["rows"] for i in range(n_clips)]
        counts = np.array([len(r) for r in rows])
        cat = np.concatenate(rows)
        out = out or ckpt.parent / f"rollouts_{ckpt.stem}.npz"
        np.savez_compressed(
            out, names=np.array(names), start=np.concatenate([[0], np.cumsum(counts)[:-1]]), count=counts,
            clip_frames=np.array([per_clip[i]["frames"] for i in range(n_clips)]), failed=np.array([per_clip[i]["failed"] for i in range(n_clips)]),
            fail_step=np.array([per_clip[i]["fail_step"] for i in range(n_clips)]), ref_frame=cat[:, -1].astype(np.int32),
            root_pos=cat[:, 0:3].astype(np.float32), root_quat=cat[:, 3:7].astype(np.float32), joint_pos=cat[:, 7 : 7 + spec.nj].astype(np.float32),
            joint_names=np.array(spec.joint_names),
        )
        print(f"{n_clips} clips, {int(sum(per_clip[i]['failed'] for i in range(n_clips)))} failed, {len(cat)} samples -> {out}")
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
