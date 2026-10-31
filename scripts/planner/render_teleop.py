"""Joystick demo video: scripted key presses drive the planner and the tracker; the frame shows the
keys, the stick, the robot and, beside it, the planner's own output.

The keys follow ``DEMO`` (forward at full speed, turn while walking, turn in place, sideways both
ways, backwards, stop), pressed through the same ``keys.Keys`` and ``TeleopSession`` as the live
teleop. Each control step records the simulated robot and the reference frame the tracker was given
(the planner's output). The planner's frames are local (velocities, not positions), so they are
integrated over a sliding ``--anchor-s`` window from where the robot was at its start
(``anchored_path``): the blue robot moves as the planner says and stays beside the silver one instead
of drifting off over the whole video. It is drawn ``--offset`` m to the camera's left, so both are seen
whole; the dots ahead of it are its current plan, 1 s. The bottom-left panel is the live teleop's
(``hud.joystick_panel``); the robot's velocity there is a 0.5 s mean.

  MUJOCO_GL=egl uv run python scripts/planner/render_teleop.py --out out/showcase/joystick.gif
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import tyro

from supermimic.planner import keys as K
from supermimic.planner.features import pose_from_features, yaw_of
from supermimic.planner.hud import ACCENT, composite, joystick_panel, legend
from supermimic.planner.rover import HOME_VEC

ROOT = Path(__file__).resolve().parents[2]

# (seconds, held keys): K.UP etc. with "move" (turn) or "strafe" (Shift + Left / Right).
DEMO = (
    (0.8, {}),
    (2.6, {K.UP: "move"}),
    (2.2, {K.UP: "move", K.LEFT: "move"}),
    (2.0, {K.RIGHT: "move"}),
    (2.0, {K.RIGHT: "strafe"}),
    (2.0, {K.LEFT: "strafe"}),
    (2.2, {K.DOWN: "move"}),
    (1.6, {}),
)


def anchored_path(ref: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World poses of the planner frames ``ref`` (T, 20): frame k is placed by the plan's own motion
    from frame k - ``window`` on, starting from the robot's pose at k - ``window``."""
    from supermimic.planner.features import quat_from_zyx

    pp, pq, pj = pose_from_features(ref, HOME_VEC)  # plan-integrated: only differences over the window are used
    pyaw, ryaw = np.unwrap(yaw_of(pq)), np.unwrap(yaw_of(root_quat))
    j = np.maximum(np.arange(len(ref)) - window, 0)
    d = pp[:, :2] - pp[j, :2]
    c, s = np.cos(ryaw[j] - pyaw[j]), np.sin(ryaw[j] - pyaw[j])
    xy = root_pos[j, :2] + np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], -1)
    yaw = ryaw[j] + pyaw - pyaw[j]
    g = ref[:, 13:16] / np.linalg.norm(ref[:, 13:16], axis=1, keepdims=True)  # gravity in the root frame
    quat = quat_from_zyx(yaw, np.arcsin(np.clip(g[:, 0], -1, 1)), np.arctan2(-g[:, 1], -g[:, 2]))
    return np.c_[xy, pp[:, 2]], quat, pj


def record(script, level: int, planner_run: str, ckpt: Path, device: str) -> dict:
    from supermimic.planner.session import TeleopSession

    s = TeleopSession(planner_run, ckpt=ckpt, device=device)
    rec: dict[str, list] = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "ref", "plan", "lit", "stick", "vel")}
    for seconds, held in script:
        K.KEYS.set_held(held)
        for _ in range(round(seconds * 50)):
            s.step(K.stick_target(K.KEYS, level, *s.box))
            p, q, j = s.robot_pose()
            rec["root_pos"].append(p)
            rec["root_quat"].append(q)
            rec["joint_pos"].append(j)
            rec["ref"].append(s.reference())
            rec["plan"].append(s.upcoming(50))
            rec["lit"].append(",".join(sorted(K.KEYS.lit())))
            rec["stick"].append(s.cmd.copy())
            rec["vel"].append(s.robot_velocity())
    s.close()
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["level"] = np.float64(K.LEVELS[level])
    return out


def scene_model(width: int, height: int):
    """``render_rollouts.scene_model``: silver robot, blue planner copy (group 4), floor, key light."""
    import importlib.util

    spec_ = importlib.util.spec_from_file_location("render_rollouts", ROOT / "scripts" / "render_rollouts.py")
    rr = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(rr)
    from supermimic.robots import get_robot_spec

    spec = get_robot_spec("rover")
    return rr.scene_model(Path(spec.xml_path), spec.root_body, width, height), rr


def main(out: Path = ROOT / "out" / "showcase" / "joystick.gif", planner_run: str = "rover_v2_l1", ckpt: Path = ROOT / "logs" / "rover_r2" / "ckpt_latest.pt",
         level: int = 3, size: str = "640x360", fps: int = 25, azimuth: float = 135.0, elevation: float = -18.0, distance: float = 2.5, offset: float = 0.9,
         ghost_alpha: float = 0.6, anchor_s: float = 0.5, recording: Path | None = None, device: str = "cuda:0") -> None:
    """``recording``: an npz written by an earlier run (``<out>.npz``), to re-render without simulating."""
    rec_path = recording or out.with_suffix(".npz")
    if recording is None:
        rec = record(DEMO, level, planner_run, ckpt, device)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(rec_path, **rec)
    rec = dict(np.load(rec_path))
    T = len(rec["root_pos"])
    pp, pq, pj = anchored_path(rec["ref"], rec["root_pos"], rec["root_quat"], round(anchor_s * 50))
    drift = np.linalg.norm(pp[:, :2] - rec["root_pos"][:, :2], axis=1)
    print(f"{T} frames; anchored planner path vs robot: {drift.mean():.3f} m mean, {drift.max():.3f} m max apart")
    vel = np.stack([np.convolve(rec["vel"][:, i], np.ones(25) / 25)[: T] for i in range(3)], 1)  # trailing 0.5 s mean
    vel[:24] = np.cumsum(rec["vel"][:24], 0) / np.arange(1, 25)[:, None]

    w, h = (int(x) for x in size.split("x"))
    model, rr = scene_model(w, h)
    nq = 7 + pj.shape[1]
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
    a = math.radians(azimuth)
    side = offset * np.array([-math.sin(a), math.cos(a), 0.0])  # the camera's left, in the world
    scene_opt, ghost_opt = rr._options(1, 2), rr._options(rr.GHOST_GROUP)
    leg = legend([("planner output", ACCENT), ("robot (physics)", (204, 207, 222))])
    assert 50 % fps == 0, "fps must divide the 50 Hz control rate (the video plays in real time)"
    step = 50 // fps
    look, frames = None, []
    with mujoco.Renderer(model, h, w) as r:
        r.update_scene(data, cam, scene_option=scene_opt)
        r.render()  # warm-up: the first frame of a fresh context comes out without materials
        for k in range(0, T, step):
            data.qpos[0:3], data.qpos[3:7], data.qpos[7:nq] = rec["root_pos"][k], rec["root_quat"][k], rec["joint_pos"][k]
            data.qpos[nq : nq + 3] = pp[k] + side
            data.qpos[nq + 3 : nq + 7] = pq[k]
            data.qpos[nq + 7 :] = pj[k]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_camlight(model, data)
            target = np.r_[0.5 * (rec["root_pos"][k, :2] + pp[k, :2] + side[:2]), 0.22]
            look = target if look is None else 0.6 * look + 0.4 * target
            cam.lookat[:] = look
            img, depth = rr._rgb_depth(r, data, cam, scene_opt)
            # The plan ahead: dots on the floor under the blue robot, 0.1 s apart.
            fp, _, _ = pose_from_features(rec["plan"][k], HOME_VEC, xy0=pp[k, :2], yaw0=float(yaw_of(pq[k])))
            r.update_scene(data, cam, scene_option=ghost_opt)
            for p in fp[5::5]:
                if r.scene.ngeom >= r.scene.maxgeom:
                    break
                mujoco.mjv_initGeom(r.scene.geoms[r.scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.018, 0, 0]), np.r_[p[:2] + side[:2], 0.02],
                                    np.eye(3).flatten(), np.array([*np.array(ACCENT) / 255, 1.0], dtype=np.float32))
                r.scene.ngeom += 1
            g_rgb = r.render()
            r.enable_depth_rendering()
            g_depth = r.render()
            r.disable_depth_rendering()
            show = (g_depth < g_depth.max() - 1e-3) & (g_depth < depth - 1e-3)
            frame = img.astype(np.float32)
            frame[show] = (1 - ghost_alpha) * frame[show] + ghost_alpha * g_rgb[show]
            frame = frame.astype(np.uint8)
            lit = set(filter(None, str(rec["lit"][k]).split(",")))
            panel = joystick_panel(lit, rec["stick"][k], vel[k], float(rec["level"]), scale=0.85)
            composite(frame, panel, 10, h - panel.shape[0] - 10)
            composite(frame, leg, 10, 10)
            frames.append(frame)
    rr.write(np.stack(frames), out, fps)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    tyro.cli(main)
