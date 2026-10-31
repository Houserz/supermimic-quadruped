"""Render recorded policy rollouts on top of their reference: the policy as the solid silver robot,
the reference clip as a blue ghost at the same place. The ghost shows only where it leaves the
robot (where it is in front of the robot's surface), so perfect tracking is a plain silver robot and
every blue patch is a tracking error. Input is ``record_rollouts.py``'s npz.

Each clip is ``NAME[@START[+SECONDS]]`` (seconds into the clip); the output is a GIF or an MP4 by
extension. Needs ffmpeg; on a headless machine set MUJOCO_GL=egl.

  MUJOCO_GL=egl uv run python scripts/render_rollouts.py --rollouts logs/rover/rollouts_ckpt_latest.npz \
      --out-dir out/showcase --ext gif tencent_dog_zig_walk_002_s16 ai4a_D1_058_KAN01_001_s16+5
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco
import numpy as np
import tyro

ROBOT_RGBA = (0.80, 0.81, 0.87, 1.0)  # silver, a little cool
FOOT_RGBA = (0.10, 0.10, 0.12, 1.0)  # black foot pads
GHOST_RGBA = (0.18, 0.48, 0.98, 1.0)
FLOOR_RGB = (0.93, 0.93, 0.94, 0.84, 0.84, 0.86)  # the two checker colours
GHOST_GROUP = 4  # the ghost's visual meshes; the robot's are group 1, the floor group 2
GHOST_ALPHA = 0.5  # blend of the ghost over the image where it is visible
GHOST_EPS = 0.025  # m: the ghost shows only where it is this much in front of the robot's surface


def _paint(spec: mujoco.MjSpec, rgba, foot_rgba, group: int) -> None:
    """Visual meshes in ``rgba`` with a glossy material; the foot spheres (collision geoms named
    ``*_foot_collision``) made visible in ``foot_rgba``; everything drawn in ``group``."""
    mat = spec.add_material(name="body")
    mat.specular, mat.shininess, mat.emission = 0.6, 0.7, 0.25
    for g in spec.geoms:
        if g.group == 1:
            g.rgba, g.material, g.group = rgba, "body", group
        elif g.name.endswith("_foot_collision"):
            g.rgba, g.group = foot_rgba, group


def scene_model(xml: Path, root_body: str, width: int, height: int, robot_rgba=ROBOT_RGBA, floor=FLOOR_RGB) -> mujoco.MjModel:
    """The robot (visual meshes and feet), a ghost copy of it, a checker floor and a sky."""
    spec = mujoco.MjSpec.from_file(str(xml))
    _paint(spec, robot_rgba, FOOT_RGBA, 1)
    ghost = mujoco.MjSpec.from_file(str(xml))
    _paint(ghost, GHOST_RGBA, GHOST_RGBA, GHOST_GROUP)
    for g in ghost.geoms:
        g.contype = g.conaffinity = 0
    spec.worldbody.add_frame().attach_body(ghost.body(root_body), "ref_", "")
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=list(floor[:3]), rgb2=list(floor[3:]), width=512, height=512)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.95, 0.96, 0.98], rgb2=[0.78, 0.82, 0.88], width=512, height=512)
    mat = spec.add_material(name="grid")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [10, 10]
    mat.emission = 0.2
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 40, 0.1], material="grid", group=2, contype=0, conaffinity=0)
    # Key light that follows the robot (its shadow stays under it however far it runs), plus a soft fill.
    spec.body(root_body).add_light(mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, pos=[0.6, 0.4, 3.0], dir=[-0.25, -0.15, -1],
                                   diffuse=[0.4, 0.4, 0.4], specular=[0.3, 0.3, 0.3], castshadow=True,
                                   type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.diffuse = [0.2, 0.2, 0.2]
    spec.visual.headlight.specular = [0.2, 0.2, 0.2]
    spec.visual.global_.offwidth = max(width, 640)
    spec.visual.global_.offheight = max(height, 480)
    spec.visual.quality.shadowsize = 4096
    return spec.compile()


def parse(item: str) -> tuple[str, float, float | None]:
    name, start, dur = item, 0.0, None
    if "+" in name:
        name, d = name.rsplit("+", 1)
        dur = float(d)
    if "@" in name:
        name, s = name.rsplit("@", 1)
        start = float(s)
    return name, start, dur


def _options(*groups: int) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in groups:
        opt.geomgroup[g] = 1
    return opt


def _rgb_depth(r: mujoco.Renderer, data, cam, opt) -> tuple[np.ndarray, np.ndarray]:
    r.update_scene(data, cam, scene_option=opt)
    rgb = r.render()
    r.enable_depth_rendering()
    depth = r.render()
    r.disable_depth_rendering()
    return rgb, depth


def render_clip(model: mujoco.MjModel, ro, idx: int, ref: dict, start_s: float, seconds: float | None, width: int, height: int,
                fps_out: int, azimuth: float, distance: float) -> np.ndarray:
    a, n = int(ro["start"][idx]), int(ro["count"][idx])
    k0 = int(round(start_s * 50))
    k1 = n if seconds is None else min(n, k0 + int(round(seconds * 50)))
    step = max(1, round(50 / fps_out))
    nq_robot = 7 + ro["joint_pos"].shape[1]
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = distance, -16.0, azimuth
    scene, ghost_only = _options(1, 2), _options(GHOST_GROUP)
    look = None
    frames = []
    with mujoco.Renderer(model, height, width) as r:
        r.update_scene(data, cam, scene_option=scene)
        r.render()  # warm-up: the first frame of a fresh context comes out without materials
        for k in range(k0, k1, step):
            j = a + k
            f = min(int(ro["ref_frame"][j]) - 1, len(ref["joint_pos"]) - 1)  # sample k <-> reference frame k
            data.qpos[0:3], data.qpos[3:7], data.qpos[7:nq_robot] = ro["root_pos"][j], ro["root_quat"][j], ro["joint_pos"][j]
            data.qpos[nq_robot : nq_robot + 3] = ref["root_pos"][f]
            data.qpos[nq_robot + 3 : nq_robot + 7] = ref["root_quat"][f]
            data.qpos[nq_robot + 7 :] = ref["joint_pos"][f]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_camlight(model, data)  # moves the robot-tracking light
            target = np.array([data.qpos[0], data.qpos[1], 0.22])
            look = target if look is None else 0.5 * look + 0.5 * target  # lightly smoothed follow, fixed azimuth
            cam.lookat[:] = look
            img, depth = _rgb_depth(r, data, cam, scene)
            g_rgb, g_depth = _rgb_depth(r, data, cam, ghost_only)
            show = (g_depth < g_depth.max() - 1e-3) & (g_depth < depth - GHOST_EPS)
            out = img.astype(np.float32)
            out[show] = (1 - GHOST_ALPHA) * out[show] + GHOST_ALPHA * g_rgb[show]
            frames.append(out.astype(np.uint8))
    return np.stack(frames)


def write(frames: np.ndarray, path: Path, fps: int) -> None:
    h, w = frames.shape[1:3]
    src = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if path.suffix == ".gif":
        vf = "split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
        cmd = src + ["-vf", vf, "-loop", "0", str(path)]
    else:
        cmd = src + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)


def main(clips: tyro.conf.Positional[list[str]], rollouts: Path, npz_dir: Path = Path("data/npz/rover"), robot: str = "rover", out_dir: Path = Path("out/showcase"),
         ext: str = "gif", size: str = "640x360", fps: int = 25, azimuth: float = 135.0, distance: float = 1.6,
         robot_rgba: tuple[float, float, float, float] = ROBOT_RGBA,
         floor: tuple[float, float, float, float, float, float] = FLOOR_RGB) -> None:
    from supermimic.robots import get_robot_spec

    spec = get_robot_spec(robot)
    w, h = (int(x) for x in size.split("x"))
    model = scene_model(Path(spec.xml_path), spec.root_body, w, h, robot_rgba, floor)
    ro = np.load(rollouts)
    names = [str(n) for n in ro["names"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in clips:
        name, start, dur = parse(item)
        d = np.load(npz_dir / f"{name}.npz")
        ri = [str(b) for b in d["body_names"]].index(spec.root_body)
        ref = {"root_pos": d["body_pos_w"][:, ri], "root_quat": d["body_quat_w"][:, ri], "joint_pos": d["joint_pos"]}
        frames = render_clip(model, ro, names.index(name), ref, start, dur, w, h, fps, azimuth, distance)
        path = out_dir / f"{name}.{ext}"
        write(frames, path, fps)
        print(f"wrote {path} ({len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    tyro.cli(main)
