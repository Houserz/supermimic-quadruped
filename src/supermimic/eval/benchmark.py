"""Benchmark evaluation (paper Appendix F.1 / plan 5.1), robot-generic.

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction fixed),
deterministic physics. A clip *fails* if a training termination (root height / tilt vs
reference, support feet when enabled) fires before the clip ends. Normalised return (paper §4 /
App. F) = sum of the per-step rewards collected over the episode / the maximum that sum could be
over the *whole clip* (max step reward * clip length), so an early fall scores low. The ratio
over the steps actually survived is kept as ``norm_return_survived`` for diagnosis only
(review 2026-10-27: the upstream code normalised by survived steps and over-rated falls).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.robots import get_robot_spec


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float  # sum of rewards / (max step reward * frames): paper protocol, falls score low
    failed: bool
    fail_step: int  # -1 if not failed
    norm_return_survived: float | None = None  # same sum / (max step reward * steps survived): diagnostic only
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)
    drift_xy: float | None = None  # mean root xy error over the episode (m), plan phase 6 root drift
    drift_yaw: float | None = None  # mean |yaw error| (rad)


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = ck["config"]["policy"]
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    ck["config"].setdefault("robot", "g1")
    # A checkpoint carries its actuator configuration: evaluate / play it at the kp and action scale it
    # was trained with (stiffness / action_scale overrides, None = robot spec).
    from supermimic.robots import set_robot_overrides

    set_robot_overrides(stiffness=ck["config"]["env"].get("stiffness"), action_scale=ck["config"]["env"].get("action_scale"))
    return policy, ck["config"]


@torch.no_grad()
def run_benchmark(
    policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True,
    anchor_obs: bool = False, robot: str = "g1", support_foot_threshold: float | None = None,
    root_height_threshold: float | None = None, root_tilt_threshold: float | None = None,
) -> list[ClipResult]:
    from supermimic.kinematics.rotations import heading_yaw, wrap_to_pi

    num_envs = min(num_envs or 1024, len(npz_files))
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        robot=robot,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
        support_foot_threshold=support_foot_threshold,
        root_height_threshold=root_height_threshold,  # None = robot spec; a run trained with other thresholds is scored with them
        root_tilt_threshold=root_tilt_threshold,
    )
    env = make_env(args, device=device)
    try:
        spec = get_robot_spec(robot)
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        max_r = TrackingRewardCfg.from_robot(spec, dt=spec.sim.control_dt).max_step_reward

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        drift_xy = torch.zeros(num_envs, device=device)
        drift_yaw = torch.zeros(num_envs, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            drift_xy += (cmd.root_pos_w[:, :2] - cmd.robot_root_pos_w[:, :2]).norm(dim=-1)
            drift_yaw += wrap_to_pi(heading_yaw(cmd.robot_root_quat_w) - heading_yaw(cmd.root_quat_w)).abs()
            done = (term | tout) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        length = int(lib.clips[ci].length)
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=length,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(length, 1)),
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            norm_return_survived=float(ret_sum[e]) / (max_r * max(st, 1)),
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                            drift_xy=float(drift_xy[e]) / max(st, 1),
                            drift_yaw=float(drift_yaw[e]) / max(st, 1),
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                steps[ids] = 0
                drift_xy[ids] = 0.0
                drift_yaw[ids] = 0.0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return (paper Fig. 3a uses the benchmark aggregate): weight = clip length in
    # frames, i.e. the total reward collected over all clips / the total maximum possible.
    tot_steps = sum(r.steps for r in results)
    tot_frames = sum(r.frames for r in results)
    w_ret = sum(r.norm_return * r.frames for r in results) / max(tot_frames, 1)
    out = {"clips": n, "failed": failed, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret}
    if results and results[0].norm_return_survived is not None:
        out["mean_norm_return_survived"] = sum(r.norm_return_survived for r in results) / max(n, 1)
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    if results and results[0].drift_xy is not None:
        out["mean_drift_xy"] = sum(r.drift_xy * r.steps for r in results) / max(tot_steps, 1)
        out["mean_drift_yaw"] = sum(r.drift_yaw * r.steps for r in results) / max(tot_steps, 1)
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))
