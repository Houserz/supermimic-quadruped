"""Render clips from the library to mp4 for a look (kinematic playback, no physics).

The camera follows the trunk; the floor is a checker plane at z = 0. Uses MuJoCo's offscreen
renderer (set MUJOCO_GL=egl on a headless machine) and ffmpeg.

  MUJOCO_GL=egl uv run python scripts/data/render_preview.py ai4a_D1_055_KAN01_001_s0 cd2_hiFive_LFleg_posCtrl
  MUJOCO_GL=egl uv run python scripts/data/render_preview.py --sampler 3   # 3 s of one clip per category
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from quadmotion import robot
from quadmotion.clip import load_clip

ROOT = Path(__file__).resolve().parents[2]
# Set by main() for the chosen robot.
NPZ = ROOT / "data" / "npz" / "rover"
OUT = NPZ / "preview"
ROOT_BODY = "link_trunk"


def scene_model(xml: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(xml))
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.93, 0.94, 0.95], rgb2=[0.80, 0.82, 0.85], width=512, height=512)
    del tex
    mat = spec.add_material(name="grid")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [8, 8]
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], material="grid", contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    return spec.compile()


def render(name: str, model: mujoco.MjModel, seconds: float | None, width: int, height: int) -> np.ndarray:
    c = load_clip(NPZ / f"{name}.npz")
    ri = c.body_names.index(ROOT_BODY)
    n = c.frames if seconds is None else min(c.frames, int(seconds * c.fps))
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = 1.6, -15.0
    frames = np.zeros((n, height, width, 3), np.uint8)
    with mujoco.Renderer(model, height, width) as r:
        for k in range(n):
            data.qpos[:3] = c.body_pos_w[k, ri]
            data.qpos[3:7] = c.body_quat_w[k, ri]
            data.qpos[7:] = c.joint_pos[k]
            mujoco.mj_kinematics(model, data)
            q = data.qpos[3:7]
            yaw = np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
            cam.azimuth = yaw + 120.0  # front-left three-quarter view, following the heading
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.22]
            r.update_scene(data, cam)
            frames[k] = r.render()
    return frames


def write_mp4(frames: np.ndarray, path: Path, fps: float = 50.0) -> None:
    h, w = frames.shape[1:3]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--robot", default="rover", help="configs/robots/<robot>.yaml")
    ap.add_argument("--sampler", type=float, default=0.0, help="seconds per category for one montage of every category")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--size", default="640x360")
    args = ap.parse_args()
    w, h = (int(x) for x in args.size.split("x"))
    global NPZ, OUT, ROOT_BODY
    quad = robot.load_robot(args.robot)
    NPZ, ROOT_BODY = ROOT / "data" / "npz" / quad.name, quad.root_body
    OUT = NPZ / "preview"
    model = scene_model(quad.xml)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.clips:
        write_mp4(render(name, model, args.seconds, w, h), OUT / f"{name}.mp4")
        print("wrote", (OUT / f"{name}.mp4").relative_to(ROOT))
    if args.sampler:
        rows = [r for r in csv.DictReader((NPZ / "manifest.csv").open()) if not r["mirror_of"]]
        by_cat: dict[str, dict] = {}
        for r in rows:  # the longest clip of each category
            if float(r["duration_s"]) > float(by_cat.get(r["category"], {"duration_s": 0})["duration_s"]):
                by_cat[r["category"]] = r
        parts = [render(r["name"], model, args.sampler, w, h) for r in by_cat.values()]
        write_mp4(np.concatenate(parts), OUT / "sampler.mp4")
        listing = "\n".join(f"{i * args.sampler:5.1f}s  {c:14s} {r['name']}" for i, (c, r) in enumerate(by_cat.items()))
        (OUT / "sampler.txt").write_text(listing + "\n")
        print(listing)
        print("wrote", (OUT / "sampler.mp4").relative_to(ROOT))


if __name__ == "__main__":
    main()
