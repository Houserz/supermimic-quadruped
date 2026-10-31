"""Training groups and sampling weights in ``data/npz/<robot>/manifest.csv`` and ``clips.yaml``.

* **holdout** -- whole raw takes, every segment cut from them and every mirror, so nothing in
  training is the same recording as a held-out clip (segments of one take, or a clip and its
  mirror, would leak). About 10 % of each source's minutes, covering its categories:

  - AI4Animation 2.11 of 23.06 min: D1_053_KAN01_001 (walk, trot, turn, idle),
    D1_ex03_KAN02_012 (walk, trot), D1_013_KAN01_001 (posture), D1_047z_KAN01_004 (walk, turn);
  - Tencent 0.88 of 8.45 min: dog_star_walk_003 (trot), dog_hit_001 (walk, turn, trot);
  - CyberDog2: swing_hip_posCtrl (its twin swing_head, the same sway about the other end of
    the trunk, stays in training).

  (Minutes on the Rover X1; another robot keeps the same takes.)

* **train / hard** -- the open-loop screen's label (``scripts/data/screen_clips.py``), kept for
  the rest; ``train`` for every clip if the screen has not been run.
* **weight** -- per-clip sampling multiplier: ``hard`` clips get ``--hard-weight`` (0.5). On the
  Rover they are mostly trots, which fell open-loop for lack of balance feedback, not for being
  infeasible, so they are trained on, at half the rate.

Run after ``screen_clips.py``; re-running either is safe (the screen keeps holdout rows).

  uv run python scripts/data/assign_groups.py --robot rover
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_TAKES = {
    "ai4animation_dog": ("D1_053_KAN01_001", "D1_ex03_KAN02_012", "D1_013_KAN01_001", "D1_047z_KAN01_004"),
    "tencent_labrador": ("dog_star_walk_003", "dog_hit_001"),
    "cyberdog2": ("cyberdog2_swing_hip_posCtrl",),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", default="rover", help="library under data/npz/<robot>")
    ap.add_argument("--hard-weight", type=float, default=0.5)
    args = ap.parse_args()
    lib = ROOT / "data" / "npz" / args.robot
    manifest, clips_yaml = lib / "manifest.csv", lib / "clips.yaml"
    cfg = yaml.safe_load(clips_yaml.read_text())
    by_name = {c["name"]: c for c in cfg["clips"]}
    rows = list(csv.DictReader(manifest.open()))
    fields = list(rows[0].keys())

    def take_of(name: str) -> tuple[str, str]:
        c = by_name[name]
        base = by_name[c["mirror_of"]] if c.get("mirror_of") else c
        return base["source"], base.get("tag", base["name"])

    found = {(s, t): 0 for s, ts in HOLDOUT_TAKES.items() for t in ts}
    for r in rows:
        c = by_name[r["name"]]
        screen = c.get("screen_group") or (r["group"] if r["group"] in ("train", "hard") else c.get("group", "train"))
        c["screen_group"] = screen
        key = take_of(r["name"])
        if key in found:
            found[key] += 1
            r["group"] = c["group"] = "holdout"
            r["weight"] = ""
        else:
            r["group"] = c["group"] = screen
            w = args.hard_weight if screen == "hard" else 1.0
            r["weight"] = f"{w:g}" if w != 1.0 else ""
        c["weight"] = float(r["weight"]) if r["weight"] else 1.0
    missing = [k for k, n in found.items() if n == 0]
    if missing:
        print(f"note: holdout takes with no clip in this library (all segments dropped by the checks): {missing}")
    cfg["groups"] = {"holdout_takes": {s: list(t) for s, t in HOLDOUT_TAKES.items()}, "hard_weight": args.hard_weight,
                     "script": "scripts/data/assign_groups.py"}
    clips_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=140))
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    agg: dict[str, list] = {}
    for r in rows:
        a = agg.setdefault(r["group"], [0, 0.0])
        a[0] += 1
        a[1] += int(r["frames"]) / 50 / 60
    for g, (n, m) in sorted(agg.items()):
        print(f"{g:8s} {n:4d} clips {m:6.2f} min (mirrors included)")


if __name__ == "__main__":
    main()
