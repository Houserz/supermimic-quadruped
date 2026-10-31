"""Build a robot's motion library from ``data/raw/``: retarget, check, cut, export, mirror.

Outputs, under ``data/npz/<robot>/`` (all regenerated on every run):

* ``<clip>.npz``  mjlab motion layout at 50 Hz, the format ``supermimic`` trains on
  (joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w,
  joint_names, body_names, fps), one file per kept clip and one per mirror;
* ``manifest.csv``  one row per kept clip (group, category, source, weight, ...);
* ``clips.yaml``  every raw clip and every candidate segment with what was done to it and, for
  the dropped ones, why;
* ``report.md``  totals per source, per check and per category.

  uv run python scripts/data/build_dataset.py --robot quad
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from quadmotion import cyberdog, dog, robot
from quadmotion.clip import FPS, body_states_from_pose, froude_time_factor, mirror_clip, trim_bounds

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
# Set by main() for the chosen robot.
OUT = ROOT / "data" / "npz" / "quad"
CLIPS_YAML = OUT / "clips.yaml"
REPORT = OUT / "report.md"
ROOT_BODY = "link_trunk"

DOG_SOURCES = {
    "ai4animation_dog": (dog.AI4ANIMATION, "data/raw/ai4animation_dog/bvh/*.bvh", "ai4a"),
    "tencent_labrador": (dog.TENCENT, "data/raw/tencent_labrador/*.bvh", "tencent"),
}
CYBERDOG_DIR = RAW / "cyberdog2" / "motion_list" / "preinstalled"
CYBERDOG_AFTER = {  # continuation -> the motion list it continues from
    "cyberdog2_sitDown_posCtrl_left": "cyberdog2_sitDown_posCtrl",
    "cyberdog2_sitDown_posCtrl_right": "cyberdog2_sitDown_posCtrl",
    "cyberdog2_sitDown_posCtrl_shake": "cyberdog2_sitDown_posCtrl",
}
CYBERDOG_CATEGORY = {"hiFive": "gesture", "sitDown": "sit", "swing": "pose", "bodystretch": "pose", "up_down": "pose"}
AMPLITUDES = (1.0, 0.9, 0.8, 0.7, 0.6)
MIN_SEGMENT_S = 2.0  # a cut of a dog clip shorter than this is dropped
MIN_CLIP_S = 1.0  # a whole hand-made clip shorter than this is dropped
KEEP_STATIC_S = 1.0  # leading / trailing holds trimmed to this


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def normalise(root_pos: np.ndarray, root_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start at the origin facing +x; height and the rest of the trajectory unchanged."""
    rot = Rotation.from_quat(root_quat, scalar_first=True)
    fwd = rot[0].apply([1.0, 0.0, 0.0])
    yaw0 = float(np.arctan2(fwd[1], fwd[0]))
    undo = Rotation.from_euler("z", -yaw0)
    pos = root_pos - [root_pos[0, 0], root_pos[0, 1], 0.0]
    return undo.apply(pos), (undo * rot).as_quat(scalar_first=True)


def export(quad: robot.Quadruped, root_pos, root_quat, q, min_s: float):
    """Trim long holds, normalise the start, FK. Returns (ClipArrays, trimmed frames) or (None, reason)."""
    qd = np.zeros_like(q)
    qd[:-1] = np.diff(q, axis=0) * FPS
    a, b = trim_bounds(qd, FPS, KEEP_STATIC_S)
    if (b - a) / FPS < min_s:
        return None, f"{(b - a) / FPS:.2f} s after trimming holds (< {min_s} s)"
    rp, rq = normalise(root_pos[a:b], root_quat[a:b])
    return body_states_from_pose(quad.model, rp, rq, q[a:b]), (a, len(q) - b)


def stats(clip, quad: robot.Quadruped) -> dict:
    ri = clip.body_names.index(ROOT_BODY)
    rpy = Rotation.from_quat(clip.body_quat_w[:, ri], scalar_first=True).as_euler("xyz")
    v = np.linalg.norm(clip.body_lin_vel_w[:, ri, :2], axis=-1)
    return {
        "root_z_range_m": [round(float(clip.body_pos_w[:, ri, 2].min()), 3), round(float(clip.body_pos_w[:, ri, 2].max()), 3)],
        "root_speed_mean_mps": round(float(v.mean()), 3),
        "root_speed_max_mps": round(float(v.max()), 3),
        "tilt_max_deg": round(float(np.degrees(np.abs(rpy[:, :2]).max())), 1),
        "joint_speed_max_radps": round(float(np.abs(clip.joint_vel).max()), 2),
        "travel_m": round(float(np.linalg.norm(clip.body_pos_w[-1, ri, :2] - clip.body_pos_w[0, ri, :2])), 2),
    }


def classify(clip, contacts: np.ndarray | None, nominal_z: float) -> str:
    """Coarse automatic label for dog-mocap segments (no labels ship with AI4Animation)."""
    ri = clip.body_names.index(ROOT_BODY)
    rot = Rotation.from_quat(clip.body_quat_w[:, ri], scalar_first=True)
    v_w = clip.body_lin_vel_w[:, ri]
    v_b = rot.inv().apply(v_w)
    speed = float(np.linalg.norm(v_w[:, :2], axis=-1).mean())
    fwd = float(v_b[:, 0].mean())
    yaw_rate = float(np.abs(clip.body_ang_vel_w[:, ri, 2]).mean())
    z = clip.body_pos_w[:, ri, 2]
    flight = float((~contacts.any(1)).mean()) if contacts is not None else 0.0
    if flight > 0.03 and z.max() > nominal_z + 0.08:
        return "jump"
    if speed < 0.1:
        return "posture" if z.max() - z.min() > 0.08 else ("turn_in_place" if yaw_rate > 0.4 else "idle")
    if fwd < -0.1:
        return "backward"
    if speed < 0.6:
        return "turn" if yaw_rate > 0.6 else "walk"
    return "trot" if speed < 1.4 else "run"


def save(clip, name: str) -> str:
    path = OUT / f"{name}.npz"
    np.savez_compressed(path, **clip.to_npz_dict())
    return str(path.relative_to(ROOT))


def edge_outlier(clip) -> tuple[bool, bool]:
    """(start, end): is the first / last step of the clip's root an outlier against the next six
    steps -- horizontal speed above their median + 0.5 m/s and above twice it, or yaw rate above
    their median + 2 rad/s. Evaluated on the float32 values the npz will hold. A benchmark or an
    episode that starts on such a step hands the robot a velocity the motion does not have (the
    Tencent BVHs' stale first frames kicked the robot 0.5-0.8 rad off its heading)."""
    ri = clip.body_names.index(ROOT_BODY)
    p = clip.body_pos_w[:, ri].astype(np.float32)
    q = clip.body_quat_w[:, ri].astype(np.float32)
    out = []
    for sl, i0 in ((slice(0, 8), 0), (slice(-8, None), -1)):
        pp, qq = p[sl], q[sl]
        w, x, y, z = qq.T
        yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        dp = np.linalg.norm(np.diff(pp[:, :2], axis=0), axis=1) * clip.fps
        dy = np.abs(np.diff(yaw)) * clip.fps
        rest = slice(1, None) if i0 == 0 else slice(None, -1)
        sp, sp_ref, y_e, y_ref = dp[i0], np.median(dp[rest]), dy[i0], np.median(dy[rest])
        out.append(bool((sp > sp_ref + 0.5 and sp > 2 * sp_ref) or (y_e > y_ref + 2.0)))
    return out[0], out[1]


def build_dog(source: str, quad: robot.Quadruped, mirror: bool, entries: list, raw_entries: list, cal_out: dict) -> None:
    skel, pattern, prefix = DOG_SOURCES[source]
    paths = sorted(ROOT.glob(pattern))
    kps = [dog.load_keypoints(p, skel) for p in paths]
    cal = dog.calibrate(kps, skel, quad.nominal_root_height())
    tf = froude_time_factor(cal.scale)
    cal_out[source] = {
        "scale": round(cal.scale, 4), "time_factor": round(tf, 4), "dog_nominal_height_m": round(cal.dog_height, 4),
        "pitch_bias_deg": round(float(np.degrees(cal.pitch_bias)), 2), "skeleton": {"anchors": skel.anchors, "paws": skel.paws},
    }
    print(f"{source}: {len(paths)} clips, scale {cal.scale:.3f}, time x{tf:.3f}, pitch bias {np.degrees(cal.pitch_bias):.1f} deg")
    for path, k in zip(paths, kps):
        r = dog.retarget(k, skel, cal, quad, checks=dog.checks_for(quad))
        valid = r.valid
        raw = {"source": source, "file": str(path.relative_to(ROOT)), "sha256": sha256(path), "src_fps": k.src_fps,
               "src_frames": k.src_frames, "robot_frames": int(len(valid)), "valid_frames": int(valid.sum()),
               "failures_frames": {n: int(m.sum()) for n, m in r.fail.items() if m.any()}, "segments": []}
        for si, (a, b) in enumerate(runs(valid)):
            name = f"{prefix}_{k.name}_s{si}"
            e = {"name": name, "source": source, "raw": raw["file"], "segment_frames": [a, b], "category": None,
                 "origin": "mocap_retarget", "tag": k.name, "keep": False}
            # Frames dropped only as stale source frames do not count against the minimum length,
            # so the rule does not make a segment disappear (two Tencent idles are exactly 2.00 s
            # with their stale head and 1.94 s without).
            stale = r.fail.get("stale_source")
            a0 = a
            while stale is not None and a0 > 0 and stale[a0 - 1] and not any(m[a0 - 1] for n, m in r.fail.items() if n != "stale_source"):
                a0 -= 1
            min_s = MIN_SEGMENT_S - (a - a0) / FPS
            if (b - a) / FPS < min_s - 1e-9:
                e["reason"] = f"segment {(b - a) / FPS:.2f} s (< {MIN_SEGMENT_S} s)"
                entries.append(e)
                continue
            clip, info = export(quad, r.root_pos[a:b], r.root_quat[a:b], r.joint_pos[a:b], min_s - 1e-9)
            edge = [0, 0]
            while clip is not None:  # step past edge outliers (edge_outlier), one frame at a time
                bad_start, bad_end = edge_outlier(clip)
                if not (bad_start or bad_end):
                    break
                lead, trail = info
                a, b = a + lead, b - trail  # the exported clip's own frames
                if bad_start:
                    a, edge[0] = a + 1, edge[0] + 1
                if bad_end:
                    b, edge[1] = b - 1, edge[1] + 1
                clip, info = export(quad, r.root_pos[a:b], r.root_quat[a:b], r.joint_pos[a:b], min_s - 1e-9)
            if clip is None:
                e["reason"] = info
                entries.append(e)
                continue
            if a > a0:
                e["stale_head_frames"] = a - a0
            if any(edge):
                e["edge_trim_frames"] = edge
                e["segment_frames"] = [a, b]
            lead, trail = info
            contacts = r.contacts[a + lead : b - trail]
            e.update(keep=True, frames=clip.frames, duration_s=round(clip.frames / FPS, 2), trimmed_frames=[lead, trail],
                     category=classify(clip, contacts, quad.nominal_root_height()),
                     ik_residual_max_mm=round(float(r.ik_residual[a:b].max() * 1000), 1), npz=save(clip, name), **stats(clip, quad))
            if mirror:
                m = mirror_clip(clip, quad)
                e["mirror"] = f"{name}_mirror"
                entries.append(e)
                entries.append({"name": f"{name}_mirror", "mirror_of": name, "source": source, "keep": True, "category": e["category"],
                                "origin": "mocap_retarget", "frames": m.frames, "duration_s": e["duration_s"], "npz": save(m, f"{name}_mirror")})
            else:
                entries.append(e)
            raw["segments"].append(name)
        raw_entries.append(raw)


def build_cyberdog(quad: robot.Quadruped, mirror: bool, entries: list, raw_entries: list, cal_out: dict) -> None:
    from quadmotion.dog import checks_for, frame_checks

    s = cyberdog.scale_for(quad)
    cal_out["cyberdog2"] = {"scale": round(s, 4), "time_factor": round(froude_time_factor(s), 4),
                            "interpreted_step_types": sorted(cyberdog.KINEMATIC), "continuations": CYBERDOG_AFTER}
    print(f"cyberdog2: scale {s:.3f}, time x{froude_time_factor(s):.3f}")
    for path in sorted(CYBERDOG_DIR.glob("*.toml")):
        name = path.stem
        after = CYBERDOG_DIR / f"{CYBERDOG_AFTER[name]}.toml" if name in CYBERDOG_AFTER else None
        raw = {"source": "cyberdog2", "file": str(path.relative_to(ROOT)), "sha256": sha256(path), "segments": []}
        if after is not None:
            raw["played_after"] = str(after.relative_to(ROOT))
        e = {"name": f"cd2_{name.removeprefix('cyberdog2_')}", "source": "cyberdog2", "raw": raw["file"], "origin": "motion_list_replay",
             "tag": name, "category": next((c for k, c in CYBERDOG_CATEGORY.items() if k in name), "pose"), "keep": False}
        try:
            cyberdog.interpret(path, quad, after=after)
        except cyberdog.UnsupportedStep as exc:
            e["reason"] = str(exc)
            raw["skipped"] = str(exc)
            entries.append(e)
            raw_entries.append(raw)
            continue
        chosen = None
        for amp in AMPLITUDES:
            m = cyberdog.interpret(path, quad, amplitude=amp, after=after)
            q, resid, clip_amt = robot.solve_ik(quad, {l: m.feet_b[:, i] for i, l in enumerate(robot.LEGS)})
            R = Rotation.from_quat(m.root_quat, scalar_first=True).as_matrix()
            fail = frame_checks(quad, m.root_pos, R, q, resid, clip_amt, checks_for(quad))
            failed = {n: int(v.sum()) for n, v in fail.items() if v.any()}
            raw.setdefault("amplitude_search", []).append({"amplitude": amp, "failures_frames": failed})
            if not failed:
                chosen = (amp, m, q, resid)
                break
        if chosen is None:
            e["reason"] = f"fails the checks at every amplitude down to {AMPLITUDES[-1]}"
            entries.append(e)
            raw_entries.append(raw)
            continue
        amp, m, q, resid = chosen
        raw["robot_frames"] = raw["valid_frames"] = int(len(q))
        clip, info = export(quad, m.root_pos, m.root_quat, q, MIN_CLIP_S)
        if clip is None:
            e["reason"] = info
            entries.append(e)
            raw_entries.append(raw)
            continue
        e.update(keep=True, amplitude=amp, frames=clip.frames, duration_s=round(clip.frames / FPS, 2), trimmed_frames=list(info),
                 support_foot_slide_mm=round(m.slide * 1000, 1), ik_residual_max_mm=round(float(resid.max() * 1000), 1),
                 npz=save(clip, e["name"]), **stats(clip, quad))
        raw["segments"].append(e["name"])
        entries.append(e)
        if mirror:
            mc = mirror_clip(clip, quad)
            e["mirror"] = f"{e['name']}_mirror"
            entries.append({"name": e["mirror"], "mirror_of": e["name"], "source": "cyberdog2", "keep": True, "category": e["category"],
                            "origin": "motion_list_replay", "frames": mc.frames, "duration_s": e["duration_s"], "npz": save(mc, e["mirror"])})
        raw_entries.append(raw)


def write_manifest(entries: list) -> None:
    with (OUT / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "keep", "group", "category", "source", "origin", "frames", "duration_s", "mirror_of", "replay_return", "weight"])
        for e in entries:
            if e.get("keep"):
                w.writerow([e["name"], 1, "train", e["category"], e["source"], e["origin"], e["frames"], e["duration_s"], e.get("mirror_of", ""), "", ""])


def write_report(entries: list, raw_entries: list, cal: dict) -> None:
    kept = [e for e in entries if e.get("keep")]
    lines = [f"# {OUT.name} motion library — build report", "", f"Generated {dt.date.today().isoformat()} by `scripts/data/build_dataset.py`.", ""]
    lines += ["## Per source", "", "| source | raw clips | robot-time min | valid min | kept clips | kept min | + mirrors min |", "|---|---|---|---|---|---|---|"]
    for src in list(DOG_SOURCES) + ["cyberdog2"]:
        raws = [r for r in raw_entries if r["source"] == src]
        rmin = sum(r.get("robot_frames", 0) for r in raws) / FPS / 60
        vmin = sum(r.get("valid_frames", 0) for r in raws) / FPS / 60
        k = [e for e in kept if e["source"] == src and not e.get("mirror_of")]
        km = sum(e["frames"] for e in k) / FPS / 60
        mm = sum(e["frames"] for e in kept if e["source"] == src) / FPS / 60
        lines.append(f"| {src} | {len(raws)} | {rmin:.1f} | {vmin:.1f} | {len(k)} | {km:.2f} | {mm:.2f} |")
    tot = sum(e["frames"] for e in kept) / FPS / 60
    lines += ["", f"**Total kept: {len(kept)} clips, {tot:.1f} min (mirrors included).**", ""]
    lines += ["## Frames failing each check (dog mocap, robot time)", "", "| check | ai4animation_dog | tencent_labrador |", "|---|---|---|"]
    names = sorted({n for r in raw_entries for n in r.get("failures_frames", {})})
    for n in names:
        cells = [str(sum(r.get("failures_frames", {}).get(n, 0) for r in raw_entries if r["source"] == s)) for s in DOG_SOURCES]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "## Kept minutes per category (without mirrors)", "", "| category | clips | min |", "|---|---|---|"]
    cats: dict[str, list] = {}
    for e in kept:
        if not e.get("mirror_of"):
            cats.setdefault(e["category"], []).append(e)
    for c, es in sorted(cats.items(), key=lambda kv: -sum(e["frames"] for e in kv[1])):
        lines.append(f"| {c} | {len(es)} | {sum(e['frames'] for e in es) / FPS / 60:.2f} |")
    lines += ["", "## CyberDog2 motion lists", "", "| file | result |", "|---|---|"]
    for r in raw_entries:
        if r["source"] != "cyberdog2":
            continue
        res = r.get("skipped") or (f"kept as {r['segments'][0]}" if r["segments"] else "dropped (see clips.yaml)")
        lines.append(f"| {Path(r['file']).name} | {res} |")
    lines += ["", "## Calibration", "", "```yaml", yaml.safe_dump(cal, sort_keys=False, allow_unicode=True).rstrip(), "```", ""]
    REPORT.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", default="rover", help="configs/robots/<robot>.yaml")
    ap.add_argument("--sources", nargs="*", default=list(DOG_SOURCES) + ["cyberdog2"])
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()
    global OUT, CLIPS_YAML, REPORT, ROOT_BODY
    quad = robot.load_robot(args.robot)
    OUT = ROOT / "data" / "npz" / quad.name
    CLIPS_YAML, REPORT, ROOT_BODY = OUT / "clips.yaml", OUT / "report.md", quad.root_body
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.npz"):
        old.unlink()
    entries: list = []
    raw_entries: list = []
    cal: dict = {}
    for src in args.sources:
        if src == "cyberdog2":
            build_cyberdog(quad, not args.no_mirror, entries, raw_entries, cal)
        else:
            build_dog(src, quad, not args.no_mirror, entries, raw_entries, cal)
    header = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "robot": {"name": quad.name, "xml": str(quad.xml.relative_to(ROOT)) if quad.xml.is_relative_to(ROOT) else str(quad.xml),
                  "xml_sha256": sha256(quad.xml), "nominal_root_height_m": round(quad.nominal_root_height(), 4),
                  "knee_sign": quad.knee_sign, "joint_names": list(quad.joint_names), "fps": FPS},
        "checks": dog.checks_for(quad).__dict__,
        "calibration": cal,
        "raw_clips": raw_entries,
        "clips": entries,
    }
    CLIPS_YAML.write_text(yaml.safe_dump(header, sort_keys=False, allow_unicode=True, width=140))
    write_manifest(entries)
    write_report(entries, raw_entries, cal)
    kept = [e for e in entries if e.get("keep")]
    print(f"kept {len(kept)} clips ({sum(e['frames'] for e in kept) / FPS / 60:.1f} min incl. mirrors) -> {OUT.relative_to(ROOT)}")
    print(f"report: {REPORT.relative_to(ROOT)}, clips: {CLIPS_YAML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
