"""Watch a trained tracking policy replay reference clips in an interactive viewer.

  uv run python scripts/play.py --ckpt logs/rover/ckpt_latest.pt --npz-dir data/npz/rover --clips tencent_dog_zig_walk_002_s16
  uv run python scripts/play.py --ckpt logs/e3/ckpt_latest.pt --npz-dir data/npz/lafan1 --num-envs 4

The grey/blue translucent robot is the reference pose; the solid robot is the policy.
Deterministic evaluation settings (no noise, no DR); every reset starts the clip from its first frame.
Native MuJoCo viewer when a display is available, otherwise viser (browser, http://localhost:8080).
Keys (native): space pause, right-arrow single step, +/- speed, backspace reset.
--no-terminations disables the root height / tilt / support-foot resets, e.g. to watch a get-up policy recover.
--agent zero replays the reference with the policy replaced by the home pose (no checkpoint needed).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint


class _DictPolicy:
    """Adapts policy(obs_policy, obs_goal) to the viewer's policy(obs) protocol."""

    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs["policy"], obs["goal"])


class _ZeroPolicy:
    def __init__(self, nj):
        self.nj = nj

    @torch.no_grad()
    def __call__(self, obs):
        return torch.zeros(obs["policy"].shape[0], self.nj, device=obs["policy"].device)


def main(
    ckpt: Path | None = None,
    robot: str = "rover",
    npz_dir: Path = Path("data/npz/rover"),
    clips: list[str] | None = None,
    num_envs: int = 1,
    device: str = "cuda:0",
    viewer: Literal["auto", "native", "viser"] = "auto",
    ghost: bool = True,
    sampling: Literal["start", "uniform"] = "start",
    no_terminations: bool = False,
    agent: Literal["trained", "zero"] = "trained",
) -> None:
    from supermimic.robots import get_robot_spec

    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    if agent == "trained":
        assert ckpt is not None, "--ckpt is required unless --agent zero"
        policy, cfg = load_policy_from_checkpoint(ckpt, device)
        robot = cfg.get("robot", robot)
        future_frames = cfg["env"]["future_frames"]
        anchor_obs = cfg["env"].get("anchor_obs", False)
    else:
        policy, future_frames, anchor_obs = None, 10, False
    spec = get_robot_spec(robot)
    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        robot=robot,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode=sampling,
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
        debug_vis=ghost,
        **({"root_height_threshold": 1.0e3, "root_tilt_threshold": 1.0e3, "support_foot_threshold": 0.0} if no_terminations else {}),
    )
    env = RslRlVecEnvWrapper(make_env(args, device=device))
    pol = _DictPolicy(policy) if policy is not None else _ZeroPolicy(spec.nj)
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    print(f"[INFO] robot {robot}: {len(files)} clip(s), {num_envs} env(s), viewer={viewer}")
    try:
        if viewer == "native":
            NativeMujocoViewer(env, pol).run()
        else:
            ViserPlayViewer(env, pol).run()
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
