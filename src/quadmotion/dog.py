"""Dog motion capture (BVH) -> a 12-DoF quadruped robot's root pose + joint angles.

Per clip:

1. Keypoints. Four *anchors* (shoulder and hip joints) and four *paws* from the skeleton, z-up,
   metres, resampled to 50 Hz and lightly smoothed (Savitzky-Golay, 0.1 s).
2. Ground. Each paw's ground level is its 3rd-percentile height in the clip; the floor is the
   median of the four, and each paw keeps a constant offset to it (a paw joint that sits a few
   cm above its pad still reads 0 when planted).
3. Body frame. x from the hip midpoint to the shoulder midpoint, y from right to left anchors,
   origin at the midpoint of the four anchors. A per-source pitch bias (dogs stand slightly
   nose-down) is removed so that the robot's neutral stance is level.
4. Scale. ``s = robot nominal root height / dog nominal anchor height`` per source. Each foot
   target is the robot's own thigh joint plus ``s`` times the dog's anchor-to-paw vector, so the
   dog's longer or shorter back does not stretch the robot's legs.
5. Contacts. A paw is planted when it is within ``contact_height`` of the ground and moving
   slower than ``contact_speed`` (both at dog scale). Planted feet are locked in x-y to the
   median of the stance and put on the ground; swing feet are kept above it.
6. Time. The clip is replayed ``sqrt(s)`` times as slow (``clip.froude_time_factor``), so that
   jumps and flight phases stay consistent with gravity at the robot's size.
7. IK (``quadmotion.robot``), then per-frame checks; the build cuts clips into the runs of frames
   that pass every check (``Checks``) and drops runs shorter than its minimum length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from quadmotion import bvh
from quadmotion.clip import FPS, froude_time_factor, resample
from quadmotion.robot import LEGS, Quadruped, solve_ik

Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # proper rotation, det +1


@dataclass(frozen=True)
class DogSkeleton:
    source: str
    anchors: dict[str, str]
    paws: dict[str, str]
    units: float = 0.01  # BVH centimetres -> metres
    contact_height: float = 0.03  # m, dog scale
    contact_speed: float = 0.35  # m/s, dog scale


AI4ANIMATION = DogSkeleton(
    "ai4animation_dog",
    anchors={"FL": "LeftArm", "FR": "RightArm", "RL": "LeftUpLeg", "RR": "RightUpLeg"},
    paws={"FL": "LeftHand_end", "FR": "RightHand_end", "RL": "LeftFoot_end", "RR": "RightFoot_end"},
)
# 3ds Max biped skeleton: the *Nub joints are helper bones (the finger nub is 23 cm long and goes
# through the floor), so the front paws are the finger joints. The hind toe joint rolls 0-3 cm up
# and forward during stance (planted in only 31% of frames of a walk, against 57% for the front
# paws), while the pad bone below it, b_*Toe002, holds still (51%; stance speed p90 0.14 m/s
# against 0.16-0.31 for b_*Toe, *Toe0Nub and b_*Ankle; dog_quad_walk_001/002, quad_run_001,
# star_walk_001).
TENCENT = DogSkeleton(
    "tencent_labrador",
    anchors={"FL": "b_LeftArm", "FR": "b_RightArm", "RL": "b_LeftLegUpper", "RR": "b_RightLegUpper"},
    paws={"FL": "b__LeftFinger", "FR": "b_RightFinger", "RL": "b_LeftToe002", "RR": "b_RightToe002"},
)
SKELETONS = {s.source: s for s in (AI4ANIMATION, TENCENT)}


@dataclass
class Keypoints:
    """50 Hz keypoints of one clip, z-up metres, floor at z = 0 for every paw's contact point."""

    name: str
    anchors: np.ndarray  # (T, 4, 3) in LEGS order
    paws: np.ndarray  # (T, 4, 3)
    paw_offset: np.ndarray  # (4,) constant height of each paw joint above its contact point
    src_fps: float
    src_frames: int

    @property
    def frames(self) -> int:
        return int(self.anchors.shape[0])


def load_keypoints(path: Path, skel: DogSkeleton) -> Keypoints:
    b = bvh.read(path)
    pos, _ = bvh.forward_kinematics(b)
    pos = pos * skel.units @ Y_UP_TO_Z_UP.T
    ia = [b.index(skel.anchors[l]) for l in LEGS]
    ip = [b.index(skel.paws[l]) for l in LEGS]
    t_src = np.arange(pos.shape[0]) / b.fps
    t_out = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / FPS)
    pts = pos[:, ia + ip]  # (T, 8, 3)
    res = np.empty((len(t_out), 8, 3))
    for j in range(8):
        for a in range(3):
            res[:, j, a] = np.interp(t_out, t_src, pts[:, j, a])
    if len(t_out) >= 7:
        res = savgol_filter(res, 5, 2, axis=0)
    anchors, paws = res[:, :4], res[:, 4:]
    ground = np.percentile(paws[:, :, 2], 3, axis=0)  # per paw
    floor = float(np.median(ground))
    offset = ground - floor
    anchors = anchors - [0, 0, floor]
    paws = paws - [0, 0, floor]
    paws[:, :, 2] -= offset  # contact point of each paw at z = 0 when planted
    return Keypoints(Path(path).stem, anchors, paws, offset, b.fps, pos.shape[0])


def body_frame(anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Origin (T, 3) and rotation matrices (T, 3, 3) of the dog's trunk from its four anchors."""
    fl, fr, rl, rr = (anchors[:, i] for i in range(4))
    front, rear = 0.5 * (fl + fr), 0.5 * (rl + rr)
    x = front - rear
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    y = 0.5 * ((fl - fr) + (rl - rr))
    y -= np.sum(y * x, -1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    R = np.stack([x, y, z], -1)
    return 0.5 * (front + rear), R


def contact_mask(paws: np.ndarray, skel: DogSkeleton) -> np.ndarray:
    """(T, 4) planted flags, debounced: gaps of <= 2 frames are filled, stances < 3 frames dropped."""
    h = paws[:, :, 2]
    # Speed as the smaller of the two one-sided differences: a gallop stance lasts 3-5 frames at
    # 50 Hz and a central difference would mix the neighbouring swing into it.
    step = np.linalg.norm(np.diff(paws[:, :, :2], axis=0), axis=-1) * FPS  # (T-1, 4)
    v = np.full_like(h, np.inf)
    v[:-1] = step
    v[1:] = np.minimum(v[1:], step)
    c = (h < skel.contact_height) & (v < skel.contact_speed)
    for i in range(4):
        c[:, i] = _close_open(c[:, i], gap=2, min_len=3)
    return c


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _close_open(m: np.ndarray, gap: int, min_len: int) -> np.ndarray:
    m = m.copy()
    for a, b in _runs(~m):
        if 0 < a and b < len(m) and b - a <= gap:
            m[a:b] = True
    for a, b in _runs(m):
        if b - a < min_len:
            m[a:b] = False
    return m


@dataclass(frozen=True)
class SourceCalibration:
    scale: float  # robot / dog
    pitch_bias: float  # rad, dog's median nose-down pitch removed from the robot's trunk
    dog_height: float  # m, nominal anchor-midpoint height of the dog
    robot_height: float  # m, nominal root height of the robot


def calibrate(clips: list[Keypoints], skel: DogSkeleton, robot_height: float) -> SourceCalibration:
    """Nominal height and pitch of the dog over the frames where it stands on (at least) three paws."""
    hs, ps = [], []
    for k in clips:
        o, R = body_frame(k.anchors)
        pitch = np.arcsin(np.clip(-R[:, 2, 0], -1, 1))
        standing = contact_mask(k.paws, skel).sum(1) >= 3
        standing &= np.abs(pitch) < np.radians(15)
        hs.append(o[standing, 2])
        ps.append(pitch[standing])
    h = float(np.median(np.concatenate(hs)))
    p = float(np.median(np.concatenate(ps)))
    return SourceCalibration(scale=robot_height / h, pitch_bias=p, dog_height=h, robot_height=robot_height)


@dataclass(frozen=True)
class Checks:
    """Per-frame acceptance; a frame failing any of these is cut out of the clip."""

    max_ik_residual: float = 0.02  # m
    max_joint_clip: float = 0.02  # rad
    max_joint_speed: float = 20.0  # rad/s, just under the Rover X1 URDF velocity limits (21-23 rad/s)
    min_trunk_corner_z: float = -0.005  # m, trunk collision box vs ground
    min_knee_z: float = 0.015  # m, knee centre (calf cylinder radius 0.02)
    max_tilt_deg: float = 60.0  # roll and pitch
    min_foot_gap: float = 0.06  # m, left-right and front-rear foot centres, trunk frame
    max_keypoint_speed: float = 12.0  # m/s at dog scale (mocap glitches)
    max_flight_s: float = 0.5  # no foot planted for longer than this = not on the floor (platforms)
    max_root_height: float = 0.75  # m
    max_root_speed: float = 3.0  # m/s horizontal; Rover X1 user guide V1.3: default max walking speed 1.8-3 m/s
    skim_height: float = 0.003  # m above resting height
    skim_speed: float = 0.5  # m/s
    # Not a check but a correction applied before IK (see ``_keep_feet_apart``): 2 x 0.05 m keeps
    # the 0.028 m foot spheres 0.044 m apart.
    min_half_track: float = 0.05


@dataclass
class Retarget:
    name: str
    root_pos: np.ndarray  # (T, 3)
    root_quat: np.ndarray  # (T, 4) wxyz
    joint_pos: np.ndarray  # (T, 12)
    contacts: np.ndarray  # (T, 4)
    fail: dict[str, np.ndarray] = field(default_factory=dict)  # check name -> (T,) bool, True = failed
    ik_residual: np.ndarray | None = None

    @property
    def valid(self) -> np.ndarray:
        bad = np.zeros(len(self.joint_pos), bool)
        for m in self.fail.values():
            bad |= m
        return ~bad


def retarget(k: Keypoints, skel: DogSkeleton, cal: SourceCalibration, robot: Quadruped, checks: Checks = Checks(), time_scaling: bool = True) -> Retarget:
    s = cal.scale
    r = robot.foot_radius
    origin, R_dog = body_frame(k.anchors)
    R = R_dog @ Rotation.from_euler("y", -cal.pitch_bias).as_matrix()
    root = s * origin + [0.0, 0.0, r]  # dog contact points at z=0 -> robot foot centres at z=r
    thigh_b = np.stack([robot.legs[l].thigh_pos for l in LEGS])  # (4, 3)
    feet = root[:, None] + np.einsum("tij,lj->tli", R, thigh_b) + s * (k.paws - k.anchors)  # (T, 4, 3)
    # Height above the ground straight from the dog's paw, not through the trunk: the robot's
    # thigh joints sit wider and further out than the dog's anchors (0.159 vs ~0.03-0.09 m to the
    # side), so any roll or pitch would otherwise lower a swing foot onto the floor and drag it
    # (dog_quad_walk_001: 1-6 frames from every stance, dog paw 1.5-3.7 cm up, robot foot on the ground).
    feet[:, :, 2] = s * k.paws[:, :, 2] + r
    feet = _keep_feet_apart(feet, root, R, checks.min_half_track)
    contacts = contact_mask(k.paws, skel)  # thresholds are at dog scale and dog time
    feet, lock_w = _lock_and_ground(feet, contacts, r)
    kp_speed = np.zeros(k.frames)
    if k.frames > 1:
        pts = np.concatenate([k.anchors, k.paws], 1)
        sp = np.linalg.norm(np.diff(pts, axis=0), axis=-1).max(1) * FPS
        kp_speed[1:] = sp
        kp_speed[0] = sp[0]
    glitch = kp_speed > checks.max_keypoint_speed
    # Robot time: the dog's clock stretched by sqrt(s) (``froude_time_factor``), resampled at 50 Hz.
    quat = Rotation.from_matrix(R).as_quat(scalar_first=True)
    tf = froude_time_factor(s) if time_scaling else 1.0
    t_dog = np.arange(k.frames) / FPS
    t_out = np.arange(0.0, t_dog[-1] * tf + 1e-9, 1.0 / FPS) / tf
    root, quat, lin, near = resample(t_dog, t_out, root, quat, {"feet": feet, "lock_w": lock_w}, {"contacts": contacts, "glitch": glitch})
    feet, contacts, lock_w = lin["feet"], near["contacts"], lin["lock_w"]
    R = Rotation.from_quat(quat, scalar_first=True).as_matrix()
    feet_b = np.einsum("tji,tlj->tli", R, feet - root[:, None])
    q, resid, clip_amt = solve_ik(robot, {l: feet_b[:, i] for i, l in enumerate(LEGS)})
    T = len(q)
    out = Retarget(k.name, root, quat, q, contacts, ik_residual=resid)
    out.fail = frame_checks(robot, root, R, q, resid, clip_amt, checks)
    out.fail["mocap_glitch"] = near["glitch"]
    out.fail["unsupported"] = _long_runs(~contacts.any(1), int(round(checks.max_flight_s * FPS))) | (root[:, 2] > checks.max_root_height)
    vxy = np.zeros(T)
    if T > 2:
        vxy[1:-1] = np.linalg.norm(root[2:, :2] - root[:-2, :2], axis=-1) * FPS / 2.0
        vxy[0], vxy[-1] = vxy[1], vxy[-2]
    out.fail["too_fast"] = vxy > checks.max_root_speed
    # A foot that was not detected as planted but sits on the ground and moves is a foot dragged
    # along the floor: the stance detector missed it, and the lock never fixed it.
    fv = np.zeros((T, 4))
    if T > 1:
        fv[1:] = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=-1) * FPS
    skim = (lock_w <= 0.0) & (feet[:, :, 2] < r + checks.skim_height) & (fv > checks.skim_speed)
    out.fail["foot_skim"] = skim.any(1)
    return out


def _long_runs(mask: np.ndarray, longer_than: int) -> np.ndarray:
    out = np.zeros_like(mask)
    for a, b in _runs(mask):
        if b - a > longer_than:
            out[a:b] = True
    return out


def _keep_feet_apart(feet: np.ndarray, root: np.ndarray, R: np.ndarray, half_track: float) -> np.ndarray:
    """Dogs cross their front paws in tight turns; the robot's legs would collide. In the trunk
    frame, left feet are kept at y >= +half_track and right feet at y <= -half_track."""
    fb = np.einsum("tji,tlj->tli", R, feet - root[:, None])
    fb[:, 0::2, 1] = np.maximum(fb[:, 0::2, 1], half_track)  # FL, RL
    fb[:, 1::2, 1] = np.minimum(fb[:, 1::2, 1], -half_track)  # FR, RR
    return root[:, None] + np.einsum("tij,tlj->tli", R, fb)


def _lock_and_ground(feet: np.ndarray, contacts: np.ndarray, r: float, blend: int = 4, clearance: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Planted feet: x-y fixed to the stance median, z = r. Swing feet: blended in and out of the
    lock over ``blend`` frames, and lifted to at least ``clearance`` above resting height outside
    that blend (scaled down inside it), because dogs scuff their paws at touch-down and lift-off
    and a robot foot copying that would drag along the floor.

    Returns the feet and the lock weight per foot (1 planted, 0 free swing)."""
    out = feet.copy()
    T = len(feet)
    weights = np.zeros((T, 4))
    for i in range(4):
        w = np.zeros(T)
        target = np.zeros((T, 3))
        for a, b in _runs(contacts[:, i]):
            p = np.median(feet[a:b, i], axis=0)
            p[2] = r
            target[a:b] = p
            w[a:b] = 1.0
            for k in range(1, blend + 1):
                wk = 1.0 - k / (blend + 1)
                for t in (a - k, b - 1 + k):
                    if 0 <= t < T and wk > w[t]:
                        w[t] = wk
                        target[t] = p
        out[:, i] = w[:, None] * target + (1 - w[:, None]) * feet[:, i]
        floor = np.where(w >= 1.0, r, r + clearance * (1.0 - w))
        out[:, i, 2] = np.maximum(out[:, i, 2], floor)
        weights[:, i] = w
    return out, weights


def frame_checks(robot: Quadruped, root: np.ndarray, R: np.ndarray, q: np.ndarray, resid: np.ndarray, clip_amt: np.ndarray, checks: Checks) -> dict[str, np.ndarray]:
    from quadmotion.robot import feet_in_trunk, knees_in_trunk, trunk_corners

    T = len(q)
    fail: dict[str, np.ndarray] = {}
    fail["ik_unreachable"] = resid.max(1) > checks.max_ik_residual
    fail["joint_limit"] = clip_amt.max(1) > checks.max_joint_clip
    dq = np.zeros(T)
    if T > 1:
        sp = np.abs(np.diff(q, axis=0)).max(1) * FPS
        dq[1:] = sp
        dq[:-1] = np.maximum(dq[:-1], sp)
    fail["joint_speed"] = dq > checks.max_joint_speed
    corners_w = root[:, None] + np.einsum("tij,cj->tci", R, trunk_corners(robot))
    fail["trunk_ground"] = corners_w[:, :, 2].min(1) < checks.min_trunk_corner_z
    knees_w = root[:, None] + np.einsum("tij,tlj->tli", R, knees_in_trunk(robot, q))
    fail["knee_ground"] = knees_w[:, :, 2].min(1) < checks.min_knee_z
    rpy = Rotation.from_matrix(R).as_euler("xyz")
    fail["tilt"] = (np.abs(rpy[:, 0]) > np.radians(checks.max_tilt_deg)) | (np.abs(rpy[:, 1]) > np.radians(checks.max_tilt_deg))
    fb = feet_in_trunk(robot, q)
    lr = np.minimum(fb[:, 0, 1] - fb[:, 1, 1], fb[:, 2, 1] - fb[:, 3, 1])  # FL-FR, RL-RR lateral
    fr = np.minimum(fb[:, 0, 0] - fb[:, 2, 0], fb[:, 1, 0] - fb[:, 3, 0])  # FL-RL, FR-RR longitudinal
    fail["feet_cross"] = (lr < checks.min_foot_gap) | (fr < checks.min_foot_gap)
    return fail


def checks_for(robot: Quadruped) -> Checks:
    """``Checks`` with the robot's overrides (``checks:`` in its ``configs/robots/*.yaml``)."""
    return Checks(**robot.checks)
