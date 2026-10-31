"""Drive the Rover from the keyboard: joystick -> walking planner -> SuperMimic tracker, live.

  MUJOCO_GL=glfw uv run python scripts/planner/teleop.py
  MUJOCO_GL=glfw uv run python scripts/planner/teleop.py --dr          # domain randomisation + obs noise
  uv run python scripts/planner/teleop.py --selftest                   # no window: scripted keys, printed results

Keys (in the viewer window; hold to move, release to stop): arrows walk forward / backward and turn,
Shift + Left / Right walks sideways, Page Up / Down changes the speed level (25 / 50 / 75 / 100 % of
the planner's box), Backspace resets, F6-F10 play the skills (sit, shake hands, high five, stretch,
dance); keypad equivalents in ``supermimic.planner.keys``.

The bottom-left panel shows the arrow pad with the pressed keys lit, the stick value and the robot's
measured velocity. The stick box of ``rover_v2_l1`` is in actual velocity: 1.5 m/s forward, 0.8 back,
0.5 sideways, 0.9 rad/s. Green dots: the current plan's root path, 1 s ahead, drawn from the robot.

A skill key stops the robot (the stick ramps to zero and the plan has to stand), blends into the
skill's library clip, plays it, blends back to standing and resumes walking. Stand guard (on unless
``--no-stand-hold``): once the stick is zero and the reference has stood for 0.5 s, the planner is
not called and the robot holds that standing frame until the stick moves.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro

from supermimic.planner import keys as K
from supermimic.planner.features import pose_from_features, yaw_of
from supermimic.planner.hud import as_rgb, joystick_panel
from supermimic.planner.rover import HOME_VEC
from supermimic.planner.session import TeleopSession

ROOT = Path(__file__).resolve().parents[2]


def run_selftest(s: TeleopSession, level: int) -> None:
    """Hold Up, Left, Shift+Right for 4 s each and print the robot's velocity (last sample and the last
    second's mean), release for 2 s (the stand guard should hold), then walk 2 s before each skill."""

    def run(n):
        for _ in range(n):
            s.step(K.stick_target(K.KEYS, level, *s.box))

    for key, mods, label in ((K.UP, 0, "Up"), (K.LEFT, 0, "Left"), (K.RIGHT, K.MOD_SHIFT, "Shift+Right")):
        K.KEYS.set_held({})
        K.KEYS.edge(key, True, mods)
        run(150)
        vs = []
        for _ in range(50):
            run(1)
            vs.append(s.robot_velocity())
        print(f"[selftest] {label:12s} stick {np.round(s.cmd, 2)} robot now {np.round(vs[-1], 2)}, last 1 s mean {np.round(np.mean(vs, 0), 2)}, "
              f"z {s.robot_pose()[0][2]:.3f}")
    K.KEYS.set_held({})
    run(100)
    print(f"[selftest] keys released 2 s: {s.status()}, rescues: {len(s.drv.rescues)}")
    for name in s.drv.lib.names:
        K.KEYS.set_held({K.UP: "move"})
        run(100)
        K.KEYS.set_held({})
        assert s.request(name), "the driver was busy"
        steps, worst = 0, 0.0
        while (s.drv.busy()[0] or steps == 0) and steps < 50 * 40:
            run(1)
            steps += 1
            worst = max(worst, s.tilt_deg())
        ev = s.drv.events[-1]
        print(f"[selftest] {name:10s} stop {ev['stop_frames'] / 50:.2f} s, blend in {ev['blend_in_frames'] / 50:.2f} s, done after {steps / 50:.1f} s, "
              f"max tilt {worst:.1f} deg, z {s.robot_pose()[0][2]:.3f}")


def main(planner_run: str = "rover_v2_l1", planner_ckpt: str = "planner_last.pt", ckpt: Path = ROOT / "logs" / "rover_r2" / "ckpt_latest.pt", clip: str = "rover_e009_s0", level: int = 1,
         dr: bool = False, stand_hold: bool = True, fall_tilt: float = math.radians(60), device: str = "cuda:0", selftest: bool = False) -> None:
    s = TeleopSession(planner_run, planner_ckpt, ckpt, clip=clip, dr=dr, stand_hold=stand_hold, device=device)
    print(f"[teleop] tracker {ckpt}; planner {planner_run}/{planner_ckpt}")
    if selftest:
        run_selftest(s, 3)
        s.close()
        return
    sim = s.env.sim
    mjm, mjd = sim.mj_model, sim.mj_data
    base_id = next(i for i in range(mjm.nbody) if (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i) or "").split("/")[-1] == "link_trunk")
    print(__doc__.split("A skill key")[0])
    fallen, note, slow = False, "", 1.0
    with mujoco.viewer.launch_passive(mjm, mjd, key_callback=K.key_callback, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base_id
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 2.2, -20.0, 135.0
        next_t = time.perf_counter()
        while viewer.is_running():
            for ev in K.KEYS.pop_events():
                if ev == "faster":
                    level = min(level + 1, len(K.LEVELS) - 1)
                elif ev == "slower":
                    level = max(level - 1, 0)
                elif ev == "reset":
                    s.reset()
                    fallen = False
                elif ev == "stop":
                    K.KEYS.set_held({})
                elif ev.startswith("skill:"):
                    note = "" if s.request(ev[6:]) else f"{ev[6:]} ignored: busy or not available"
            t0 = time.perf_counter()
            s.step(K.stick_target(K.KEYS, level, *s.box))
            fallen = fallen or s.tilt_deg() > math.degrees(fall_tilt)
            mjd.qpos[:] = sim.data.qpos[0].cpu().numpy()
            mjd.qvel[:] = sim.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(mjm, mjd)
            # The plan's root path, 1 s ahead, drawn from the robot (the plan is local: rotate it onto the robot's heading).
            viewer.user_scn.ngeom = 0
            rp, rq, _ = pose_from_features(s.upcoming(50), HOME_VEC)
            q = s.robot_pose()[1]
            yaw = float(yaw_of(q)) - float(yaw_of(rq[0]))
            cy, sy = math.cos(yaw), math.sin(yaw)
            base = mjd.xpos[base_id]
            for p in rp[::5, :2] - rp[0, :2]:
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                    break
                pos = np.array([base[0] + cy * p[0] - sy * p[1], base[1] + sy * p[0] + cy * p[1], 0.01])
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.015, 0, 0]), pos, np.eye(3).flatten(),
                                    np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32))
                viewer.user_scn.ngeom += 1
            panel = as_rgb(joystick_panel(K.KEYS.lit(), s.cmd, s.robot_velocity(), K.LEVELS[level]))
            viewer.set_images((mujoco.MjrRect(12, 12, panel.shape[1], panel.shape[0]), panel))
            left = "Status\nKeys\nSkills"
            right = (f"{'FALLEN - press Backspace' if fallen else s.status()}{'  ' + note if note else ''}"
                     f"{'' if slow <= 1.05 else f'  (running {slow:.1f}x slower than real time)'}\n"
                     "arrows walk / turn, Shift+L/R sideways, PgUp / PgDn speed, Backspace reset\n"
                     "F6 sit  F7 shake hands  F8 high five  F9 stretch  F10 dance")
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value, mujoco.mjtGridPos.mjGRID_TOPLEFT.value, left, right))
            viewer.sync()
            next_t += 0.02
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
                slow = 1.0
            else:
                slow = (time.perf_counter() - t0) / 0.02
                next_t = time.perf_counter()
    s.close()


if __name__ == "__main__":
    tyro.cli(main)
