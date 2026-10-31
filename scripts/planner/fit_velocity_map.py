"""Fit the walking policy's steady-state velocity map and write the actual-velocity labels.

The walking policy walks slower than its command (~0.88 vx, ~0.71 vy, ~0.92 wz), so a planner
conditioned on the command walks slower than the joystick says. The actual-velocity planners take
the robot's velocity instead, with one of two labels per frame:

* ``label_l1`` = G(cmd): G is the policy's steady-state map, command -> mean actual velocity
  (heading-frame vx, vy, world yaw rate), a polynomial without constant term (G(0) = 0) fitted on
  the steady part of every hold of the training episodes (frames at least ``--settle`` s after the
  command last changed, holds with at least one gait period of them). As smooth as the command
  and noise-free; only its units change.
* ``label_l2`` = the actual velocity itself, centered ``--window``-frame mean (0.62 s: one 0.6 s gait
  period, which removes the gait's own oscillation; centered, so it has no delay but looks 0.3 s ahead).

Also fits the inverse H(v) ~ G^-1 (same form, roles swapped): the planner's clock is the policy's,
gated on the policy's command (|cmd| < 0.1), and at deployment only the joystick value v exists,
so the gate is evaluated on H(v). Polynomial degree is chosen on the validation episodes' holds.

Writes ``<npz_dir>/../velocity_map.json`` and adds ``label_l1``, ``label_l2`` to every clip.

    uv run python scripts/planner/fit_velocity_map.py --npz-dir data/planner/rover_v2/npz
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import tyro

from supermimic.planner.features import features_from_pose
from supermimic.planner.rover import (
    ACC_CMD_V2,
    JOY_HI,
    JOY_LO,
    VelocityMap,
    monomial_exponents,
    steady_mask,
)

ROOT = Path(__file__).resolve().parents[2]


def raw_velocity(rp: np.ndarray, rq: np.ndarray) -> np.ndarray:
    """(T, 3) per-frame heading-frame vx, vy and world yaw rate (central differences, no smoothing)."""
    f = features_from_pose(rp, rq, np.zeros((len(rp), 0)), np.zeros(0))
    return np.concatenate([f[:, 4:6], f[:, 7:8]], -1)


def centered_mean(v: np.ndarray, window: int) -> np.ndarray:
    k = np.ones(window)
    num = np.stack([np.convolve(v[:, j], k, mode="same") for j in range(v.shape[1])], -1)
    return num / np.convolve(np.ones(len(v)), k, mode="same")[:, None]


def holds(cmd: np.ndarray, v: np.ndarray, settle: int, min_frames: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """(command, mean actual velocity, frames) of the steady part of every hold."""
    m = steady_mask(cmd, settle)
    out = []
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= min_frames:
            out.append((cmd[a], v[a:b].mean(0), b - a))
    return out


def main(npz_dir: Path = ROOT / "data" / "planner" / "rover_v2" / "npz", settle: float = 1.5, min_s: float = 0.6, window: int = 31, degrees: tuple[int, ...] = (1, 2, 3, 4, 5),
         val_every: int = 10, write: bool = True) -> None:
    t0 = time.time()
    files = sorted(npz_dir.glob("*.npz"))
    data = {}
    H = {"train": [], "val": []}
    for f in files:
        with np.load(f) as z:
            rp, rq, cmd = z["body_pos_w"][:, 0].astype(np.float64), z["body_quat_w"][:, 0].astype(np.float64), z["cmd"].astype(np.float64)
        v = raw_velocity(rp, rq)
        data[f] = (cmd, v)
        env = int(re.search(r"_e(\d+)_", f.stem).group(1))
        H["val" if env % val_every == val_every - 1 else "train"] += holds(cmd, v, round(settle * 50), round(min_s * 50))
    C = {k: np.stack([h[0] for h in H[k]]) for k in H}
    V = {k: np.stack([h[1] for h in H[k]]) for k in H}
    print(f"holds: train {len(C['train'])}, val {len(C['val'])} ({time.time() - t0:.0f} s)")

    def fit(x, y, deg):
        e = monomial_exponents(deg)
        phi = VelocityMap.features(x, e)
        w, *_ = np.linalg.lstsq(phi, y, rcond=None)
        return e, w

    rows = {}
    for deg in degrees:
        e, w = fit(C["train"], V["train"], deg)
        _, wi = fit(V["train"], C["train"], deg)
        vm = VelocityMap(e, w, wi)
        err = vm.actual(C["val"]) - V["val"]
        rt = vm.command(vm.actual(C["val"])) - C["val"]
        rows[deg] = {"val_rmse": np.sqrt((err ** 2).mean(0)).round(4).tolist(), "val_p95_abs": np.percentile(np.abs(err), 95, axis=0).round(4).tolist(),
                     "inverse_round_trip_rmse": np.sqrt((rt ** 2).mean(0)).round(4).tolist(), "params": int(w.size)}
        print(f"degree {deg}: val rmse vx/vy/wz {rows[deg]['val_rmse']} p95 {rows[deg]['val_p95_abs']} | H(G(c)) - c rmse {rows[deg]['inverse_round_trip_rmse']}")
    best = min(degrees, key=lambda d: sum(rows[d]["val_rmse"]))
    e, w = fit(np.concatenate([C["train"]]), np.concatenate([V["train"]]), best)
    _, wi = fit(V["train"], C["train"], best)
    vm = VelocityMap(e, w, wi)
    lin = np.array([vm.actual(np.eye(3)[i] * s)[i] / s for i, s in enumerate((0.5, 0.3, 0.5))])
    print(f"degree {best} chosen; small-command gains vx/vy/wz {lin.round(3).tolist()}")
    # The joystick's ramp cap in actual units: the recording's cap through the map's gain at the box edge.
    edge = np.array([vm.actual(np.eye(3)[i][None] * h)[0, i] / h for i, h in enumerate((1.5, 0.75, 1.0))])
    joy_acc = tuple(float(np.floor(10 * a * g) / 10) for a, g in zip(ACC_CMD_V2, np.minimum(edge, lin)))
    out = {"what": "walking policy's steady-state map, command -> actual heading velocity (fit_velocity_map.py)", "date": time.strftime("%Y-%m-%d %H:%M"), "npz_dir": str(npz_dir),
           "settle_s": settle, "min_steady_s": min_s, "holds": {"train": len(C["train"]), "val": len(C["val"])}, "degree": best, "degree_scan": rows,
           "small_command_gain": lin.round(4).tolist(), "edge_gain": edge.round(4).tolist(), "l2_window_frames": window,
           "joy_lo": list(JOY_LO), "joy_hi": list(JOY_HI), "joy_acc": list(joy_acc), "acc_cmd": list(ACC_CMD_V2), **vm.to_dict()}
    (npz_dir.parent / "velocity_map.json").write_text(json.dumps(out, indent=1))
    print(f"joystick ramp cap {joy_acc}; -> {npz_dir.parent / 'velocity_map.json'}")

    # Lag of the actual velocity behind G(cmd) on ramps (cross-correlation of the derivatives).
    lags = []
    for f, (cmd, v) in list(data.items())[:60]:
        l1 = vm.actual(cmd)
        l2 = centered_mean(v, window)
        for j in range(3):
            a, b = np.diff(l1[:, j]), np.diff(l2[:, j])
            if np.abs(a).sum() < 1e-6:
                continue
            cc = [np.dot(a[max(0, -k) : len(a) - max(0, k)], b[max(0, k) : len(b) - max(0, -k)]) for k in range(-25, 26)]
            lags.append(int(np.argmax(cc)) - 25)
    print(f"actual velocity (centered {window}-frame mean) behind G(cmd): median lag {np.median(lags):.0f} frames, p10-p90 {np.percentile(lags, 10):.0f}..{np.percentile(lags, 90):.0f}")

    if write:
        for f, (cmd, v) in data.items():
            with np.load(f) as z:
                d = {k: z[k] for k in z.files if k not in ("label_l1", "label_l2")}
            np.savez_compressed(f, **d, label_l1=vm.actual(cmd).astype(np.float32), label_l2=centered_mean(v, window).astype(np.float32))
        print(f"labels written to {len(data)} clips ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    tyro.cli(main)
