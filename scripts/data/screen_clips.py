"""Import check and open-loop feasibility screen of a library built by ``build_dataset.py``.

1. Import check: joint / body names and order against this stack's copy of the
   model, 50 Hz, finite, and every body pose and velocity re-derived by FK from the joint angles
   and root pose (1e-4 m, 1e-2 m/s). A clip failing it is not screened.
2. **Windowed** PD open-loop replay (``ReferencePolicy``: the PD set-point is the next reference
   pose), no noise, no randomisation: every ``stride_s`` a ``window_s`` window is started from the
   reference state and played open-loop; a window *fails* when a training termination fires
   (root height off by > 0.15 m, tilt off by > 60 deg). A clip is ``hard`` when more than
   ``hard_frac`` of its windows fail.

Why windows and why a stiffer servo than the robot's kp 25 (measured 2026-10-29 on this library):

* At kp 25 a standing clip sags and folds a rear leg within 0.7 s (``rover.rover_spec``), so a
  replay at the real gains scores the servo's softness: 506 / 566 whole clips fell. A policy
  supplies the holding offset; the screen asks whether the *motion* can be followed.
* Whole-clip open-loop replay of a quadruped measures drift, not feasibility: at kp 120 a 45 s
  idle clip (ai4a_D1_ex03_KAN02_003_s15) stood for 38 s while its root wandered 0.5 m from the
  reference, then tipped at a weight shift its misplaced feet could not take. Short windows
  restarted from the reference state remove the accumulated drift.
* Stiffness sweep on every 4th original clip, whole-clip replay (failed / 71): kp 25: 66,
  50: 55, 80: 50, 120: 42, 200: 46 -- 120 is the stiffest setting that still helps.

Results go back into the library: ``data/npz/<robot>/feasibility.json`` (per clip and per window),
the ``replay_return`` / ``group`` columns of ``manifest.csv`` and the ``replay_*`` / ``group`` /
``screen_group`` fields of ``clips.yaml``. Rows already in the ``holdout`` group keep it;
afterwards run ``scripts/data/assign_groups.py`` so the ``weight`` column follows the train / hard
labels. Needs the robot's training spec (``src/supermimic/robots``) and a GPU.

  uv run python scripts/data/screen_clips.py --robot rover
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml

from supermimic.data.clips import body_states_from_pose, load_clip, model_body_names
from supermimic.robots import get_robot_spec, set_robot_overrides

ROOT = Path(__file__).resolve().parents[2]
FPS = 50


class ReferencePolicy:
    """policy(obs_policy, obs_goal) -> action that sets the PD target to the next reference pose."""

    def __init__(self, spec, device):
        self.G = spec.goal_feat_dim
        self.nj = spec.nj
        self.scale = torch.tensor([spec.action_scale_dict()[j] for j in spec.joint_names], device=device)

    def __call__(self, obs_policy, obs_goal):
        return obs_goal[:, self.G : self.G + self.nj] / self.scale


def import_check(path: Path, spec, model, names_model) -> str | None:
    c = load_clip(path)
    if c.joint_names != list(spec.joint_names):
        return "joint order differs"
    if c.body_names != names_model:
        return "body order differs from the model"
    if abs(c.fps - FPS) > 1e-9:
        return f"fps {c.fps}"
    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if not np.isfinite(getattr(c, k)).all():
            return f"{k} not finite"
    ri = c.body_names.index(spec.root_body)
    rec = body_states_from_pose(model, spec, c.body_pos_w[:, ri], c.body_quat_w[:, ri], c.joint_pos, 1.0 / c.fps)
    e_pos = float(np.abs(rec.body_pos_w - c.body_pos_w).max())
    e_vel = max(float(np.abs(getattr(rec, k) - getattr(c, k)).max()) for k in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"))
    if e_pos > 1e-4 or e_vel > 1e-2:
        return f"FK mismatch pos {e_pos:.1e} vel {e_vel:.1e}"
    return None


@torch.no_grad()
def windowed_replay(spec, files: list[str], window: int, stride: int, device: str, max_envs: int) -> tuple[list[str], list[dict]]:
    """Returns the library's clip names and, per window: clip index, start frame, horizon, failed,
    fail step, normalised return (sum of rewards / (max step reward * horizon))."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.rewards import TrackingRewardCfg

    # The env is built once at the batch size; every batch re-points each env at a window start.
    probe = make_env(SuperMimicEnvCfgArgs(npz_files=files, robot=spec.name, num_envs=1, record_transitions=False), device=device)
    lib = probe.command_manager.get_term("motion").library
    names = [c.name for c in lib.clips]
    windows = []
    for ci, c in enumerate(lib.clips):
        last = c.length - 2  # the step after the last played frame must stay inside the clip
        h = min(window, last)
        for s in range(0, max(1, last - h + 1), stride):
            windows.append({"clip": ci, "start": s, "horizon": h})
    probe.close()

    n_env = min(len(windows), max_envs)
    args = SuperMimicEnvCfgArgs(
        npz_files=files, robot=spec.name, num_envs=n_env, episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False,
        reset_noise=False, domain_randomization=False, sampling_mode="start", record_transitions=False, seed=0,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        policy = ReferencePolicy(spec, device)
        max_r = TrackingRewardCfg.from_robot(spec, dt=spec.sim.control_dt).max_step_reward
        for b0 in range(0, len(windows), n_env):
            batch = windows[b0 : b0 + n_env]
            pad = batch + [batch[-1]] * (n_env - len(batch))
            start = torch.tensor([int(lib.clip_start[w["clip"]]) + w["start"] for w in pad], device=device)
            horizon = torch.tensor([w["horizon"] for w in pad], device=device)
            cmd.forced_frame[:] = start
            obs, _ = env.reset()
            ret = torch.zeros(n_env, device=device)
            done = torch.zeros(n_env, dtype=torch.bool, device=device)
            failed = torch.zeros(n_env, dtype=torch.bool, device=device)
            fail_step = torch.full((n_env,), -1, dtype=torch.long, device=device)
            for t in range(int(horizon.max())):
                obs, rew, term, tout, _ = env.step(policy(obs["policy"], obs["goal"]))
                live = ~done
                ret += torch.where(live, rew, torch.zeros_like(rew))
                newly_failed = live & term
                failed |= newly_failed
                fail_step[newly_failed] = t
                done |= term | tout | (t + 1 >= horizon)
                if done.all():
                    break
            for k, w in enumerate(batch):
                w["failed"] = bool(failed[k])
                w["fail_step"] = int(fail_step[k])
                w["norm_return"] = float(ret[k]) / (max_r * max(w["horizon"], 1))
            print(f"  windows {min(b0 + n_env, len(windows))}/{len(windows)}")
    finally:
        env.close()
    return names, windows


def main(robot: str = "rover", device: str = "cuda:0", stiffness: float = 120.0, window_s: float = 2.0, stride_s: float = 1.0, hard_frac: float = 0.2, max_envs: int = 4096) -> None:
    set_robot_overrides(stiffness=stiffness)
    spec = get_robot_spec(robot)
    model = spec.load_model()
    names_model = model_body_names(model)
    npz_dir = ROOT / "data" / "npz" / robot
    manifest_path = npz_dir / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open()))
    fields = list(rows[0].keys())

    ok_rows, import_fail = [], {}
    for r in rows:
        err = import_check(npz_dir / f"{r['name']}.npz", spec, model, names_model)
        if err:
            import_fail[r["name"]] = err
        else:
            ok_rows.append(r)
    print(f"import check: {len(ok_rows)} pass, {len(import_fail)} fail")
    for n, e in list(import_fail.items())[:10]:
        print(f"  FAIL {n}: {e}")

    files = [str(npz_dir / f"{r['name']}.npz") for r in ok_rows]
    names, windows = windowed_replay(spec, files, int(round(window_s * FPS)), int(round(stride_s * FPS)), device, max_envs)
    per_clip: dict[str, dict] = {}
    for ci, name in enumerate(names):
        ws = [w for w in windows if w["clip"] == ci]
        nf = sum(w["failed"] for w in ws)
        per_clip[name] = {
            "windows": len(ws), "failed_windows": nf, "fail_frac": nf / max(len(ws), 1),
            "mean_window_return": float(np.mean([w["norm_return"] for w in ws])),
            "failed_window_starts_s": [round(w["start"] / FPS, 2) for w in ws if w["failed"]],
        }
    settings = {"stiffness": stiffness, "window_s": window_s, "stride_s": stride_s, "hard_frac": hard_frac,
                "root_height_threshold": spec.root_height_threshold, "root_tilt_threshold": spec.root_tilt_threshold}
    (npz_dir / "feasibility.json").write_text(json.dumps({"settings": settings, "import_failed": import_fail, "clips": per_clip,
                                                          "windows": [{**w, "clip": names[w["clip"]]} for w in windows]}, indent=1))

    def group_of(name: str) -> str:
        return "hard" if per_clip[name]["fail_frac"] > hard_frac else "train"

    for r in rows:
        if r["name"] in import_fail:
            r["group"], r["replay_return"] = "import_failed", ""
            continue
        r["replay_return"] = f"{per_clip[r['name']]['mean_window_return']:.4f}"
        if r.get("group") != "holdout":  # held-out takes stay held out (scripts/data/assign_groups.py)
            r["group"] = group_of(r["name"])
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    clips_yaml = npz_dir / "clips.yaml"
    cfg = yaml.safe_load(clips_yaml.read_text())
    for c in cfg["clips"]:
        if c["name"] in import_fail:
            c["group"], c["import_check"] = "import_failed", import_fail[c["name"]]
        elif c["name"] in per_clip:
            p = per_clip[c["name"]]
            c["replay_return"] = round(p["mean_window_return"], 4)
            c["replay_fail_frac"] = round(p["fail_frac"], 3)
            c["replay_failed_window_starts_s"] = p["failed_window_starts_s"]
            c["screen_group"] = group_of(c["name"])
            if c.get("group") != "holdout":
                c["group"] = c["screen_group"]
    cfg["screening"] = {"script": "scripts/data/screen_clips.py", **settings}
    clips_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=140))

    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {"n": 0, "hard": 0, "min": 0.0, "hard_min": 0.0, "ff": [], "ret": []})
    for r in rows:
        if r["name"] in import_fail:
            continue
        p = per_clip[r["name"]]
        a = agg[(r["source"], r["category"])]
        m = int(r["frames"]) / FPS / 60
        a["n"] += 1
        a["min"] += m
        a["ff"].append(p["fail_frac"])
        a["ret"].append(p["mean_window_return"])
        if r["group"] == "hard":
            a["hard"] += 1
            a["hard_min"] += m
    print(f"\n{'source':18s} {'category':14s} {'clips':>5s} {'hard':>5s} {'min':>6s} {'hard min':>8s} {'win fail':>8s} {'return':>7s}")
    for (src, cat), a in sorted(agg.items()):
        print(f"{src:18s} {cat:14s} {a['n']:5d} {a['hard']:5d} {a['min']:6.2f} {a['hard_min']:8.2f} {np.mean(a['ff']):8.3f} {np.mean(a['ret']):7.3f}")
    n_win = len(windows)
    n_fail = sum(w["failed"] for w in windows)
    hard = [r for r in rows if r["group"] == "hard"]
    m_all = sum(int(r["frames"]) for r in rows if r["name"] not in import_fail) / FPS / 60
    m_hard = sum(int(r["frames"]) for r in hard) / FPS / 60
    print(f"\nwindows {n_win}, failed {n_fail} ({n_fail / max(n_win, 1):.1%}); hard clips {len(hard)}/{len(ok_rows)} ({m_hard:.1f} of {m_all:.1f} min); train {m_all - m_hard:.1f} min")
    print("saved", npz_dir / "feasibility.json")


if __name__ == "__main__":
    tyro.cli(main)
