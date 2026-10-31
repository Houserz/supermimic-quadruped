"""Independent check of every exported clip, re-derived from the npz alone.

Layout, names, 50 Hz, finite values, joint limits and FK re-computation of every body pose and
velocity, plus two things a tracking reference must not ask for and that FK consistency cannot
see: a foot below the ground and a planted foot sliding.

A foot counts as planted when its centre is within 1.5 mm of resting height (z = radius); its
slide is its horizontal speed while planted. The control group for that measure is the raw,
unlocked retarget (``--control``), which must show the sliding the lock removes.

  uv run python scripts/data/check_library.py --robot quad
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from quadmotion import robot
from quadmotion.clip import body_states_from_pose, load_clip

ROOT = Path(__file__).resolve().parents[2]
# Locked feet sit at exactly z = radius; swing feet near a stance are lifted 2-10 mm (dog._lock_and_ground).
PLANTED_TOL = 0.0015
KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_names", "body_names", "fps")


def foot_sites(quad: robot.Quadruped, root_pos, root_quat, q) -> np.ndarray:
    model = quad.model
    data = mujoco.MjData(model)
    sid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in quad.foot_sites]
    out = np.zeros((len(q), 4, 3))
    for k in range(len(q)):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root_pos[k], root_quat[k], q[k]
        mujoco.mj_kinematics(model, data)
        out[k] = data.site_xpos[sid]
    return out


def check(path: Path, quad: robot.Quadruped, names_model: list[str]) -> dict:
    with np.load(path) as d:
        missing = [k for k in KEYS if k not in d.files]
        assert not missing, f"missing keys {missing}"
        assert all(d[k].dtype == np.float32 for k in KEYS[:6]), "per-frame arrays must be float32"
    c = load_clip(path)
    assert c.joint_names == list(quad.joint_names), "joint order"
    assert c.body_names == names_model, "body order"
    assert abs(c.fps - 50.0) < 1e-9, f"fps {c.fps}"
    for k in KEYS[:6]:
        assert np.isfinite(getattr(c, k)).all(), f"{k} not finite"
    lo, hi = quad.q_lo - 0.01, quad.q_hi + 0.01
    assert ((c.joint_pos >= lo) & (c.joint_pos <= hi)).all(), "joint limits"
    ri = c.body_names.index(quad.root_body)
    rec = body_states_from_pose(quad.model, c.body_pos_w[:, ri], c.body_quat_w[:, ri], c.joint_pos, 1.0 / c.fps)
    e_pos = float(np.abs(rec.body_pos_w - c.body_pos_w).max())
    sign = np.sign(np.sum(rec.body_quat_w * c.body_quat_w, -1, keepdims=True))
    e_quat = float(np.abs(rec.body_quat_w * sign - c.body_quat_w).max())
    e_vel = max(float(np.abs(getattr(rec, k) - getattr(c, k)).max()) for k in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"))
    assert e_pos < 1e-4 and e_quat < 1e-4 and e_vel < 1e-2, f"FK mismatch pos {e_pos:.1e} quat {e_quat:.1e} vel {e_vel:.1e}"
    feet = foot_sites(quad, c.body_pos_w[:, ri], c.body_quat_w[:, ri], c.joint_pos)
    r = quad.foot_radius
    pen = float(max(0.0, r - feet[:, :, 2].min()))
    planted = feet[:, :, 2] < r + PLANTED_TOL
    v = np.zeros(feet.shape[:2])
    v[1:] = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=-1) * c.fps
    both = planted.copy()
    both[1:] &= planted[:-1]
    slide = v[both]
    return {"frames": c.frames, "fk_pos": e_pos, "penetration_mm": pen * 1000,
            "planted_frac": float(planted.mean()), "slide_p95_mps": float(np.percentile(slide, 95)) if slide.size else 0.0,
            "slide_max_mps": float(slide.max()) if slide.size else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", default="quad", help="configs/robots/<robot>.yaml")
    ap.add_argument("--control", action="store_true", help="also measure sliding on unlocked retargets (control group)")
    args = ap.parse_args()
    quad = robot.load_robot(args.robot)
    OUT = ROOT / "data" / "npz" / quad.name
    names_model = robot.body_names(quad.model)
    rows = list(csv.DictReader((OUT / "manifest.csv").open()))
    n_ok, bad, agg = 0, [], []
    for row in rows:
        try:
            res = check(OUT / f"{row['name']}.npz", quad, names_model)
            agg.append((row["name"], res))
            n_ok += 1
        except AssertionError as e:
            bad.append((row["name"], str(e)))
    print(f"checked {len(rows)} clips: {n_ok} pass, {len(bad)} fail")
    for n, e in bad[:20]:
        print(f"  FAIL {n}: {e}")
    pen = np.array([a["penetration_mm"] for _, a in agg])
    sp95 = np.array([a["slide_p95_mps"] for _, a in agg])
    smax = np.array([a["slide_max_mps"] for _, a in agg])
    fk = np.array([a["fk_pos"] for _, a in agg])
    print(f"FK re-computation max error {fk.max():.1e} m")
    print(f"foot penetration: max {pen.max():.2f} mm over all clips")
    print(f"planted-foot slide: per-clip p95 median {np.median(sp95):.3f} m/s, worst clip p95 {sp95.max():.3f} m/s, worst single frame {smax.max():.3f} m/s")
    worst = sorted(agg, key=lambda a: -a[1]["slide_p95_mps"])[:5]
    print("  worst p95 slide:", ", ".join(f"{n} {a['slide_p95_mps']:.3f}" for n, a in worst))
    if args.control:
        from quadmotion import dog

        skel = dog.AI4ANIMATION
        paths = sorted((ROOT / "data/raw/ai4animation_dog/bvh").glob("*.bvh"))[:8]
        kps = [dog.load_keypoints(p, skel) for p in paths]
        cal = dog.calibrate(kps, skel, quad.nominal_root_height())
        lock = dog._lock_and_ground
        dog._lock_and_ground = lambda feet, contacts, r, blend=4, clearance=0.0: (np.maximum(feet, [-np.inf, -np.inf, r]), np.zeros(feet.shape[:2]))  # no lock: control
        try:
            slides = []
            for k in kps:
                rt = dog.retarget(k, skel, cal, quad)
                feet = foot_sites(quad, rt.root_pos, rt.root_quat, rt.joint_pos)
                planted = feet[:, :, 2] < quad.foot_radius + PLANTED_TOL
                v = np.zeros(feet.shape[:2])
                v[1:] = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=-1) * 50
                both = planted.copy()
                both[1:] &= planted[:-1]
                if both.any():
                    slides.append(np.percentile(v[both], 95))
            print(f"control (no foot lock, 8 AI4Animation clips): per-clip p95 slide median {np.median(slides):.3f} m/s")
        finally:
            dog._lock_and_ground = lock


if __name__ == "__main__":
    main()
