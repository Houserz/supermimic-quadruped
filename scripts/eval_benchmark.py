"""Evaluate a checkpoint on every clip of a motion set (paper Appendix F protocol).

  uv run python scripts/eval_benchmark.py --ckpt logs/e2/ckpt_latest.pt --npz-dir data/npz/lafan1
  uv run python scripts/eval_benchmark.py --ckpt logs/rover/ckpt_latest.pt --npz-dir data/npz/rover \
      --manifest data/npz/rover/manifest.csv --group holdout
"""

from __future__ import annotations

from pathlib import Path

import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results, summarize


def main(ckpt: Path, npz_dir: Path = Path("data/npz/rover"), clips: list[str] | None = None, clip_list: Path | None = None, manifest: Path | None = None, group: str | None = None, exclude: str | None = None, num_envs: int = 1024, device: str = "cuda:0", out: Path | None = None) -> None:
    if clip_list is not None:
        clips = [l.strip() for l in open(clip_list) if l.strip()]
    files = [str(f) for f in discover_npz(npz_dir, names=clips, manifest=manifest, group=group, exclude=exclude)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    results = run_benchmark(
        policy, files, device=device, num_envs=num_envs, future_frames=cfg["env"]["future_frames"], anchor_obs=cfg["env"].get("anchor_obs", False),
        robot=cfg.get("robot", "g1"), support_foot_threshold=cfg["env"].get("support_foot_threshold"),
        root_height_threshold=cfg["env"].get("root_height_threshold"), root_tilt_threshold=cfg["env"].get("root_tilt_threshold"),
    )
    s = summarize(results)
    print(f"\n{s['clips']} clips | failed {s['failed']} | mean norm return {s['mean_norm_return']:.4f} | step-weighted {s['step_weighted_norm_return']:.4f}")
    if "mean_drift_xy" in s:
        print(f"root drift: xy {s['mean_drift_xy']:.3f} m, yaw {s['mean_drift_yaw']:.3f} rad (step-weighted means)")
    if "terms" in s:
        print("per-step term means (max dt*weight each):")
        for k, v in s["terms"].items():
            print(f"  {k:28s} {v:.4f}")
    worst = sorted(results, key=lambda r: r.norm_return)[:10]
    for r in worst:
        print(f"  {r.name:32s} ret {r.norm_return:.3f} steps {r.steps}/{r.frames} {'FAIL@' + str(r.fail_step) if r.failed else ''}")
    out = out or ckpt.parent / f"benchmark_{ckpt.stem}{('_' + group) if group else ''}.json"
    save_results(results, out)
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
