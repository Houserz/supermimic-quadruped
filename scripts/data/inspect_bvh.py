"""Print what the retargeting has to assume about a BVH source: vertical axis, units, which
points touch the ground, and whether bone lengths stay constant (they do only if the
translation convention in ``quadmotion.bvh`` is right for that file).

  uv run python scripts/data/inspect_bvh.py data/raw/ai4animation_dog/bvh/D1_001_KAN01_001.bvh
"""

from __future__ import annotations

import sys

import numpy as np

from quadmotion import bvh


def main(path: str) -> None:
    b = bvh.read(path)
    pos, _ = bvh.forward_kinematics(b)
    T, J, _ = pos.shape
    print(f"{path}: {T} frames @ {b.fps:.1f} fps, {J} joints (incl. end sites)")
    lo, hi = pos.reshape(-1, 3).min(0), pos.reshape(-1, 3).max(0)
    print(f"  extent per axis: {np.round(hi - lo, 1)}")
    # Vertical axis: the one along which the median per-frame spread is ~ body height and the min is ~constant.
    spread = np.median(pos.max(1) - pos.min(1), axis=0)
    print(f"  median per-frame spread per axis: {np.round(spread, 1)}")
    for ax in range(3):
        mins = pos[:, :, ax].min(1)
        print(f"  axis {'XYZ'[ax]}: per-frame min  p5 {np.percentile(mins, 5):8.2f}  p50 {np.percentile(mins, 50):8.2f}  p95 {np.percentile(mins, 95):8.2f}")
    # Bone length stability.
    worst = []
    for j, joint in enumerate(b.joints):
        if joint.parent < 0:
            continue
        L = np.linalg.norm(pos[:, j] - pos[:, joint.parent], axis=-1)
        if L.mean() > 1e-6:
            worst.append((float(L.std() / L.mean()), joint.name, float(L.mean()), float(np.linalg.norm(joint.offset))))
    worst.sort(reverse=True)
    print("  bone length relative std (worst 5): " + ", ".join(f"{n} {s:.1e} (mean {m:.2f}, offset {o:.2f})" for s, n, m, o in worst[:5]))
    return pos, b


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
