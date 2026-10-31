"""Rover skills: walking from the planner, skills played from the motion library, every switch through standing.

docs/PLANNER.md has the overview. In short:

* **Walking** is the walking planner's (``planner.rover.load_planner``; any run, the run name is a
  parameter), driven exactly as in ``scripts/planner/closed_loop.py``: replan every
  ``replan`` steps from the executed reference, crossfade, the tracker follows the next 10 frames.
* **Skills** are library clips the tracker was trained on, played back as they are (the planner
  never generates them). A skill is one clip, or several stitched with blends (the pose medley).
* **Switching** always goes through standing. A request first ramps the command to zero and waits
  until the plan stands (velocities, joint speeds and foot heights below thresholds), then the
  reference blends from the standing frame into the skill's first frame, plays the skill, holds its
  last frame, blends into ``hub`` (a standing frame of the planner's own data) and holds it for one
  planner context, from which the planner resumes.
* **Blends** keep the feet planted: the skill is placed (2-D rigid fit of its first frame's feet on
  the current feet), the root moves along a minimum-jerk path, the feet move linearly from where
  they are to where the skill wants them (only the stance mismatch, ~1 cm), and the joints come
  from leg IK. ``method="joint"`` (joint-space interpolation with the same root path) and
  ``"joint_noshift"`` (joint-space interpolation, root fixed: the feet slide by the stance
  difference) are kept for the comparison in the doc.

Nothing here reads the robot: the state machine sees only its own reference and the command, as
the planner does. Everything is in the planner's local features (``planner.features``), so the
tracker's goal is built by ``planner.goal.plan_goal_frames`` whatever the source of the frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from supermimic.planner.features import feat_slices, features_from_pose, quat_from_zyx, yaw_of
from supermimic.planner.goal import plan_goal_frames
from supermimic.planner.rover import (
    FOOT_BODIES,
    FOOT_GEOMS,
    HOME_VEC,
    NJ,
    RateLimiter,
    joystick_condition,
    steady_mask,
)

ROOT = Path(__file__).resolve().parents[3]
LIBRARY_DIR = ROOT / "data" / "npz" / "rover"
PLANNER_DATA = ROOT / "data" / "planner" / "rover" / "npz"
FPS = 50.0
SL = feat_slices(NJ)

# Segment codes of every reference frame (logged by the closed loop).
WALK, STOP, BLEND_IN, CLIP, HOLD, BLEND_OUT, HUB, INNER, STAND, RESCUE = range(10)
SEG_NAMES = ("walk", "stop", "blend_in", "clip", "hold", "blend_out", "hub", "inner_blend", "stand_hold", "stand_rescue")
# Driver modes.
M_WALK, M_STOP, M_SKILL = 0, 1, 2


@dataclass(frozen=True)
class SkillDef:
    """A skill: library clips played in order (blended between), then ``hold_s`` on the last frame."""

    clips: tuple[str, ...]
    label: str
    hold_s: float = 0.3
    trained: bool = True  # the tracker in use has trained on the clips; others load only with allow_untrained


# All clips are CyberDog2 motion-list replays in the tracker's train group.
SKILLS: dict[str, SkillDef] = {
    "sit": SkillDef(("cd2_sitDown_posCtrl",), "sit", hold_s=1.5),
    "shake": SkillDef(("cd2_sitDown_posCtrl_shake",), "shake hands", hold_s=0.5),
    "high_five": SkillDef(("cd2_hiFive_LFleg_posCtrl",), "high five"),
    "stretch": SkillDef(("cd2_bodystretch",), "stretch"),
    # Not dance data: a medley of the library's pose clips (head swing, body up-down, head swing
    # mirrored), stitched with blends.
    "dance": SkillDef(("cd2_swing_head_posCtrl", "cd2_up_down", "cd2_swing_head_posCtrl_mirror"), "dance (pose medley)"),
}


@dataclass
class SkillCfg:
    method: str = "ik"  # blend: "ik" (feet planted), "joint", "joint_noshift"
    qd_max: float = 1.0  # rad/s: blend duration = max joint change / qd_max ...
    hd_max: float = 0.10  # m/s: ... or height change / hd_max ...
    blend_min_s: float = 0.5  # ... clipped to [blend_min_s, blend_max_s]
    blend_max_s: float = 2.0
    hub_hold_s: float = 0.5  # standing on the hub frame before the planner resumes (its context is 0.5 s)
    accel: tuple[float, float, float] = (2.0, 1.5, 2.0)  # stick rate limit for a command-conditioned planner; a checkpoint's joy_acc wins
    ready_v: float = 0.03  # m/s: planned heading speed below this ...
    ready_wz: float = 0.08  # rad/s: ... yaw rate below this ...
    ready_qd: float = 0.4  # rad/s: ... joint speeds below this, over the next ready_window frames of the plan
    ready_window: int = 25
    ready_feet_z: float = 0.040  # m: every foot centre below this (stance 0.027-0.028) at the launch frame
    ready_replans: int = 2  # consecutive replans that must pass
    stop_timeout_s: float = 3.0  # start anyway after this long (counted as a timeout): the planner can fail to stand
    queue_s: float = 40.0  # longest reference a skill may queue
    stand_hold: bool = True  # stand guard: zero stick + the executed reference stood -> hold that frame, no replanning
    stand_rescue_s: float | None = 2.0  # stand guard: zero stick this long while walking and still not held -> blend to the hub and hold it


# ---------------------------------------------------------------- poses <-> features, one frame at a time
def minjerk(n: int) -> np.ndarray:
    """(n+1,) minimum-jerk progress 0 -> 1 (zero velocity and acceleration at both ends)."""
    s = np.linspace(0.0, 1.0, n + 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def rp_of(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ZYX pitch and roll of wxyz quaternions (as ``pose_from_features`` recovers them from gravity)."""
    w, x, y, z = np.moveaxis(quat, -1, 0)
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return pitch, roll


def pose_of_feature(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One feature frame -> (root_pos (3,), root_quat (4,), joints (12,)) at xy 0, yaw 0."""
    g = f[SL["grav"]] / np.linalg.norm(f[SL["grav"]])
    pitch = np.arcsin(np.clip(g[0], -1.0, 1.0))
    roll = np.arctan2(-g[1], -g[2])
    return np.array([0.0, 0.0, f[SL["height"]][0]]), quat_from_zyx(np.array(0.0), pitch, roll), f[SL["joint"]] + HOME_VEC


@dataclass
class Poses:
    """A pose sequence in some world frame: root_pos (T,3), root_quat (T,4) wxyz, joints (T,12), seg (T,) codes."""

    pos: np.ndarray
    quat: np.ndarray
    q: np.ndarray
    seg: np.ndarray

    def __len__(self) -> int:
        return len(self.pos)

    def cat(self, other: Poses, drop_first: bool = True) -> Poses:
        k = 1 if drop_first else 0
        return Poses(np.concatenate([self.pos, other.pos[k:]]), np.concatenate([self.quat, other.quat[k:]]), np.concatenate([self.q, other.q[k:]]), np.concatenate([self.seg, other.seg[k:]]))

    def frame(self, i: int) -> Poses:
        i = i % len(self)
        return Poses(self.pos[i : i + 1], self.quat[i : i + 1], self.q[i : i + 1], self.seg[i : i + 1])

    def hold(self, frames: int, seg: int) -> Poses:
        last = self.frame(-1)
        return Poses(np.repeat(last.pos, frames, 0), np.repeat(last.quat, frames, 0), np.repeat(last.q, frames, 0), np.full(frames, seg))

    def transformed(self, yaw: float, xy: np.ndarray) -> Poses:
        """Rotate the sequence by ``yaw`` about the world z axis, then translate by ``xy``."""
        c, s = np.cos(yaw), np.sin(yaw)
        pos = self.pos.copy()
        pos[:, 0], pos[:, 1] = c * self.pos[:, 0] - s * self.pos[:, 1] + xy[0], s * self.pos[:, 0] + c * self.pos[:, 1] + xy[1]
        qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        w1, x1, y1, z1 = qz
        w2, x2, y2, z2 = np.moveaxis(self.quat, -1, 0)
        quat = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
        return Poses(pos, quat, self.q.copy(), self.seg.copy())

    def features(self) -> np.ndarray:
        return features_from_pose(self.pos, self.quat, self.q, HOME_VEC, FPS)


def poses_from_npz(path: Path, seg: int = CLIP) -> Poses:
    with np.load(path) as z:
        assert str(z["body_names"][0]) == "link_trunk" and float(z["fps"]) == FPS, path
        return Poses(z["body_pos_w"][:, 0].astype(np.float64), z["body_quat_w"][:, 0].astype(np.float64), z["joint_pos"].astype(np.float64), np.full(len(z["joint_pos"]), seg))


# ---------------------------------------------------------------- kinematics
class LegKinematics:
    """Foot-sphere centres by MuJoCo FK and per-leg damped Gauss-Newton IK on the Rover model."""

    def __init__(self):
        import mujoco

        from supermimic.robots import get_robot_spec
        from supermimic.robots.rover import JOINT_NAMES

        self.mj = mujoco
        self.model = get_robot_spec("rover").load_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.q_adr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
        self.v_adr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
        self.geom = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOMS]
        self.body = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES]
        self.lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][0] for n in JOINT_NAMES])
        self.hi = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][1] for n in JOINT_NAMES])
        self._jacp = np.zeros((3, m.nv))

    def _set(self, pos, quat, q) -> None:
        d = self.data
        d.qpos[:] = 0.0
        d.qpos[0:3] = pos
        d.qpos[3:7] = quat
        d.qpos[self.q_adr] = q
        self.mj.mj_kinematics(self.model, d)

    def feet(self, pos: np.ndarray, quat: np.ndarray, q: np.ndarray) -> np.ndarray:
        """(4, 3) world foot-sphere centres (FL, FR, RL, RR) of one pose."""
        self._set(pos, quat, q)
        return self.data.geom_xpos[self.geom].copy()

    def feet_seq(self, p: Poses) -> np.ndarray:
        return np.stack([self.feet(p.pos[i], p.quat[i], p.q[i]) for i in range(len(p))])

    def ik(self, pos: np.ndarray, quat: np.ndarray, targets: np.ndarray, q0: np.ndarray, iters: int = 30, tol: float = 1e-6) -> tuple[np.ndarray, float]:
        """Joints putting the foot centres on ``targets`` (4, 3) for the given root pose; starts at ``q0``.
        Returns (q, worst foot error in m)."""
        mj, m, d = self.mj, self.model, self.data
        q = q0.copy()
        err = np.inf
        for _ in range(iters):
            self._set(pos, quat, q)
            mj.mj_comPos(m, d)
            e = targets - d.geom_xpos[self.geom]
            err = float(np.abs(e).max())
            if err < tol:
                break
            for leg in range(4):
                mj.mj_jac(m, d, self._jacp, None, d.geom_xpos[self.geom[leg]], self.body[leg])
                cols = self.v_adr[3 * leg : 3 * leg + 3]
                J = self._jacp[:, cols]
                dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(3), J.T @ e[leg])
                q[3 * leg : 3 * leg + 3] = np.clip(q[3 * leg : 3 * leg + 3] + dq, self.lo[3 * leg : 3 * leg + 3], self.hi[3 * leg : 3 * leg + 3])
        return q, err


def fit_rigid_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    """Yaw and translation (Kabsch in the plane) that map the xy of points ``src`` onto ``dst`` (both (k, 3))."""
    a, b = src[:, :2], dst[:, :2]
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    yaw = float(np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1]))
    c, s = np.cos(yaw), np.sin(yaw)
    return yaw, cb - np.array([c * ca[0] - s * ca[1], s * ca[0] + c * ca[1]])


def blend_poses(a: Poses, b: Poses, kin: LegKinematics, cfg: SkillCfg, seg: int) -> Poses:
    """Frames from the last frame of ``a`` to the first frame of ``b`` (both included), both already placed.

    Root: minimum-jerk in xy, yaw, height, pitch and roll. Joints: ``ik`` keeps each foot on the
    straight line from its start to its end position (minimum-jerk in time); ``joint`` and
    ``joint_noshift`` interpolate the joints.
    """
    pa, qa, ja = a.pos[-1], a.quat[-1], a.q[-1]
    pb, qb, jb = b.pos[0], b.quat[0], b.q[0]
    dur = max(np.abs(jb - ja).max() / cfg.qd_max, abs(pb[2] - pa[2]) / cfg.hd_max)
    n = round(np.clip(dur, cfg.blend_min_s, cfg.blend_max_s) * FPS)
    s = minjerk(n)[:, None]
    ya, yb = float(yaw_of(qa)), float(yaw_of(qb))
    dyaw = (yb - ya + np.pi) % (2 * np.pi) - np.pi
    pit_a, rol_a = rp_of(qa)
    pit_b, rol_b = rp_of(qb)
    pos = pa + s * (pb - pa)
    yaw = ya + s[:, 0] * dyaw
    quat = quat_from_zyx(yaw, pit_a + s[:, 0] * (pit_b - pit_a), rol_a + s[:, 0] * (rol_b - rol_a))
    q = ja + s * (jb - ja)
    if cfg.method == "ik":
        fa, fb = kin.feet(pa, qa, ja), kin.feet(pb, qb, jb)
        for i in range(1, n):
            q[i], _ = kin.ik(pos[i], quat[i], fa + s[i, 0] * (fb - fa), q[i])
    return Poses(pos, quat, q, np.full(n + 1, seg))


def place_on(seq: Poses, anchor: Poses, kin: LegKinematics, cfg: SkillCfg) -> Poses:
    """``seq`` moved so that its first frame stands where ``anchor``'s last frame stands: feet fitted
    (``ik`` / ``joint``) or root xy and yaw aligned (``joint_noshift``)."""
    if cfg.method == "joint_noshift":
        yaw = float(yaw_of(anchor.quat[-1]) - yaw_of(seq.quat[0]))
        moved = seq.transformed(yaw, np.zeros(2))
        return moved.transformed(0.0, anchor.pos[-1, :2] - moved.pos[0, :2])
    f_seq = kin.feet(seq.pos[0], seq.quat[0], seq.q[0])
    f_anchor = kin.feet(anchor.pos[-1], anchor.quat[-1], anchor.q[-1])
    yaw, xy = fit_rigid_2d(f_seq, f_anchor)
    return seq.transformed(yaw, xy)


# ---------------------------------------------------------------- the hub and the skill sequences
def hub_frame(npz_dir: Path = PLANNER_DATA, clips: int = 10, settle_s: float = 1.0) -> np.ndarray:
    """(20,) the standing frame the planner resumes from: the medoid of the planner data's steady
    standing frames (zero command held at least ``settle_s``) over its first ``clips`` validation
    clips, velocities set to zero."""
    files = [f for f in sorted(Path(npz_dir).glob("*.npz")) if int(f.stem.split("_e")[1].split("_")[0]) % 10 == 9][:clips]
    frames = []
    for f in files:
        with np.load(f) as z:
            feat = features_from_pose(z["body_pos_w"][:, 0].astype(float), z["body_quat_w"][:, 0].astype(float), z["joint_pos"].astype(float), HOME_VEC)
            cmd = np.asarray(z["cmd"], np.float64)
        m = steady_mask(cmd, int(settle_s * FPS)) & (np.abs(cmd).max(1) == 0)
        frames.append(feat[m])
    x = np.concatenate(frames)
    keep = np.r_[SL["joint"].start : SL["grav"].stop]
    med = np.median(x[:, keep], 0)
    hub = x[np.argmin(np.abs(x[:, keep] - med).sum(1))].copy()
    hub[SL["lin_h"]] = 0.0
    hub[SL["yaw_rate"]] = 0.0
    return hub


class SkillLibrary:
    """The skills' fixed parts, built once: [clip, (inner blend, clip)..., hold, blend out, hub hold] in
    the first clip's frame. Only the blend in depends on where the robot stands; ``sequence`` adds it."""

    def __init__(self, skills: dict[str, SkillDef] | None = None, cfg: SkillCfg | None = None, library_dir: Path = LIBRARY_DIR, hub: np.ndarray | None = None, kin: LegKinematics | None = None,
                 allow_untrained: bool = False):
        self.cfg = cfg or SkillCfg()
        self.defs = {k: v for k, v in (SKILLS if skills is None else skills).items() if v.trained or allow_untrained}
        self.names = list(self.defs)
        self.kin = kin or LegKinematics()
        self.hub = hub_frame() if hub is None else hub
        pos, quat, q = pose_of_feature(self.hub)
        self.hub_pose = Poses(pos[None], quat[None], q[None], np.array([HUB]))
        self.library_dir = Path(library_dir)
        self.core: dict[str, Poses] = {}
        self.clip_ranges: dict[str, list[tuple[str, int, int]]] = {}  # skill -> (clip, first, last) frame indices in core
        for name, sd in self.defs.items():
            self.core[name], self.clip_ranges[name] = self._core(sd)

    def _core(self, sd: SkillDef) -> tuple[Poses, list]:
        cfg, kin = self.cfg, self.kin
        seq = poses_from_npz(self.library_dir / f"{sd.clips[0]}.npz")
        ranges = [(sd.clips[0], 0, len(seq) - 1)]
        for name in sd.clips[1:]:
            nxt = place_on(poses_from_npz(self.library_dir / f"{name}.npz"), seq, kin, cfg)
            seq = seq.cat(blend_poses(seq, nxt, kin, cfg, INNER))
            first = len(seq) - 1
            seq = seq.cat(nxt)
            ranges.append((name, first, len(seq) - 1))
        seq = seq.cat(seq.hold(round(sd.hold_s * FPS), HOLD), drop_first=False)
        hub = place_on(self.hub_pose, seq, kin, cfg)
        seq = seq.cat(blend_poses(seq, hub, kin, cfg, BLEND_OUT))
        seq.seg[-1] = HUB
        seq = seq.cat(seq.hold(round(cfg.hub_hold_s * FPS), HUB), drop_first=False)
        return seq, ranges

    def sequence(self, name: str, start: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Features (T, 20) and segment codes (T,) from the standing feature frame ``start``
        (excluded) through the skill to the hub; ``info`` has the blend-in length and the IK error."""
        pos, quat, q = pose_of_feature(start)
        a = Poses(pos[None], quat[None], q[None], np.array([STOP]))
        core = place_on(self.core[name], a, self.kin, self.cfg)
        bi = blend_poses(a, core, self.kin, self.cfg, BLEND_IN)
        seq = bi.cat(core)
        feat = seq.features()
        return feat[1:].astype(np.float32), seq.seg[1:].copy(), {"blend_in_frames": len(bi) - 1}


# ---------------------------------------------------------------- the driver: planner + skills, n environments
class SkillDriver:
    """Turns a command and skill requests into the tracker's goal, for ``n`` environments.

    Each environment holds the executed reference (``hist``, the planner's context) and a queue of
    upcoming frames. Walking: every ``replan`` steps the planner replaces the queue (crossfading
    ``blend`` frames), as in ``closed_loop.py``. A request switches the environment to stopping (the
    command ramps to zero); when the plan stands, the skill's sequence replaces the queue; when it
    has played, walking resumes from the hub. One request at a time: a request while busy is ignored.
    """

    def __init__(self, planner, ck_planner: dict, lib: SkillLibrary, n: int, default_joint_pos: torch.Tensor, device: str, blend: int = 5, K: int = 10):
        self.planner, self.lib, self.cfg = planner, lib, lib.cfg
        self.n, self.device, self.blend, self.K = n, device, blend, K
        self.C, self.H, self.replan = planner.cfg.context, planner.cfg.horizon, int(ck_planner["replan"])
        self.ck = ck_planner  # the condition is built by planner.rover.joystick_condition from the checkpoint's own label / map
        self.accel = tuple(ck_planner.get("joy_acc", self.cfg.accel))  # the stick's slew limit, planner.rover.RateLimiter
        self.home = torch.tensor(HOME_VEC, dtype=torch.float32, device=device)
        self.default_joint_pos = default_joint_pos
        self.Q = max(int(self.cfg.queue_s * FPS), self.H + K + 1)
        self.gait = torch.zeros(n, dtype=torch.long, device=device)

    def reset(self, ctx: np.ndarray, clock0: np.ndarray) -> None:
        """Every environment walking (standing) from the context ``ctx`` (n, C, 20); planner clock from ``clock0``."""
        n, d = self.n, self.device
        self.hist = torch.tensor(ctx, dtype=torch.float32, device=d)
        self.queue = self.hist[:, -1:].repeat(1, self.Q, 1)
        self.seg = np.full((n, self.Q), WALK, np.int8)
        self.mode = np.full(n, M_WALK)
        self.pending = np.full(n, -1)
        self.skill = np.full(n, -1)  # skill being stopped for / played
        self.remaining = np.zeros(n, int)
        self.ready = np.zeros(n, int)
        self.held = np.zeros(n, bool)  # stand guard engaged
        self.zero_since = np.zeros(n, int)  # step at which the stick last became zero
        self.rescues: list[dict] = []  # stand rescues (the planner did not stand on its own)
        self.t_req = np.zeros(n, int)
        self.limiter = RateLimiter(self.accel, 1.0 / FPS, n)
        self.cmd = np.zeros((n, 3))
        self.clock0 = np.asarray(clock0)
        self.t = 0
        self.events: list[dict] = []  # one per launched skill

    def request(self, env: int, skill: str) -> bool:
        if self.mode[env] != M_WALK or self.pending[env] >= 0:
            return False
        self.pending[env] = self.lib.names.index(skill)
        return True

    def busy(self) -> np.ndarray:
        return (self.mode != M_WALK) | (self.pending >= 0)

    def _launch(self, e: int, timeout: bool) -> None:
        start = self.hist[e, -1].cpu().numpy().astype(np.float64)
        name = self.lib.names[self.skill[e]]
        feat, seg, info = self.lib.sequence(name, start)
        L = min(len(feat), self.Q)
        self.queue[e, :L] = torch.as_tensor(feat[:L], device=self.device)
        self.queue[e, L:] = self.queue[e, L - 1]
        self.seg[e, :L], self.seg[e, L:] = seg[:L], HUB
        self.remaining[e] = L
        self.mode[e] = M_SKILL
        self.events.append({"env": e, "skill": name, "t_request": int(self.t_req[e]), "t_launch": self.t, "stop_frames": self.t - int(self.t_req[e]), "timeout": timeout,
                            "blend_in_frames": info["blend_in_frames"], "frames": L})

    def _stood(self, plan: torch.Tensor) -> np.ndarray:
        """Per row of ``plan`` (b, H, 20): does the plan stand over its first ``ready_window`` frames?"""
        c = self.cfg
        w = plan[:, : c.ready_window]
        v = w[..., SL["lin_h"]][..., :2].abs().amax((1, 2))
        wz = w[..., SL["yaw_rate"]].abs().amax((1, 2))
        qd = ((w[:, 1:, SL["joint"]] - w[:, :-1, SL["joint"]]).abs() * FPS).amax((1, 2))
        return ((v < c.ready_v) & (wz < c.ready_wz) & (qd < c.ready_qd)).cpu().numpy()

    def _feet_down(self, e: int) -> bool:
        pos, quat, q = pose_of_feature(self.hist[e, -1].cpu().numpy().astype(np.float64))
        return bool((self.lib.kin.feet(pos, quat, q)[:, 2] < self.cfg.ready_feet_z).all())

    def _quiet(self, envs: np.ndarray) -> np.ndarray:
        """Has the *executed* reference of ``envs`` stood over its last ``ready_window`` frames (the stand
        test's thresholds)? Unlike ``_stood`` this does not trust the plan's next frames."""
        c = self.cfg
        w = self.hist[torch.as_tensor(envs, device=self.device), -c.ready_window :]
        v = w[..., SL["lin_h"]][..., :2].abs().amax((1, 2))
        wz = w[..., SL["yaw_rate"]].abs().amax((1, 2))
        qd = ((w[:, 1:, SL["joint"]] - w[:, :-1, SL["joint"]]).abs() * FPS).amax((1, 2))
        return ((v < c.ready_v) & (wz < c.ready_wz) & (qd < c.ready_qd)).cpu().numpy()

    def _hold(self, e: int) -> None:
        """Stand guard: from now on the reference is the last executed frame, standing still."""
        f = self.hist[e, -1].clone()
        f[SL["lin_h"]] = 0.0
        f[SL["yaw_rate"]] = 0.0
        self.queue[e] = f
        self.seg[e] = STAND
        self.held[e] = True

    def _rescue(self, e: int) -> None:
        """Stand guard, last resort: the planner has not stood within ``stand_rescue_s`` of a zero stick
        (it can fall into a ~2.4 Hz yaw oscillation in place). Blend from the
        current frame to the hub, feet planted (``blend_poses``), and hold the hub."""
        pos, quat, q = pose_of_feature(self.hist[e, -1].cpu().numpy().astype(np.float64))
        a = Poses(pos[None], quat[None], q[None], np.array([WALK]))
        hub = place_on(self.lib.hub_pose, a, self.lib.kin, self.cfg)
        seq = a.cat(blend_poses(a, hub, self.lib.kin, self.cfg, RESCUE))
        feat = seq.features()[1:].astype(np.float32)
        feat[-1, SL["lin_h"]] = 0.0
        feat[-1, SL["yaw_rate"]] = 0.0
        L = len(feat)
        self.queue[e, :L] = torch.as_tensor(feat, device=self.device)
        self.queue[e, L:] = self.queue[e, L - 1]
        self.seg[e, :L], self.seg[e, L:] = RESCUE, STAND
        self.held[e] = True
        self.rescues.append({"env": int(e), "t": self.t, "zero_for_frames": self.t - int(self.zero_since[e])})

    def _guard(self) -> None:
        """Stand guard (``SkillCfg.stand_hold``), checked on replan steps: walking or stopping environments
        with a zero stick whose executed reference stood with every foot down are held; a held
        environment that is stopping for a skill launches it at once; a non-zero stick releases the hold
        (the planner restarts at the next replan from the held, standing context)."""
        zero = np.abs(self.cmd).max(1) == 0
        self.held &= zero & (self.mode != M_SKILL)
        cand = np.flatnonzero(zero & ~self.held & (self.mode != M_SKILL))
        if len(cand):
            for e in cand[self._quiet(cand)]:
                if self._feet_down(e):
                    self._hold(e)
        if self.cfg.stand_rescue_s is not None:
            late = zero & ~self.held & (self.mode == M_WALK) & (self.t - self.zero_since >= self.cfg.stand_rescue_s * FPS)
            for e in np.flatnonzero(late):
                self._rescue(e)
        for e in np.flatnonzero(self.held & (self.mode == M_STOP)):
            self.held[e] = False
            self._launch(e, False)

    @torch.no_grad()
    def step(self, target_cmd: np.ndarray) -> torch.Tensor:
        """One control step: the goal (n, K * goal dim) for the tracker, then advance the reference."""
        c, t = self.cfg, self.t
        new_req = (self.pending >= 0) & (self.mode == M_WALK)
        for e in np.flatnonzero(new_req):
            self.mode[e], self.skill[e], self.pending[e], self.t_req[e], self.ready[e] = M_STOP, self.pending[e], -1, t, 0
        target = np.where((self.mode == M_WALK)[:, None], target_cmd, 0.0)
        moving = np.abs(self.cmd).max(1) > 0
        self.cmd = self.limiter(target)
        self.zero_since[moving & (np.abs(self.cmd).max(1) == 0)] = t
        if self.cfg.stand_hold:
            if t % self.replan == 0:
                self._guard()
            else:  # a moving stick releases the hold at once; the planner takes over at the next replan
                self.held &= np.abs(self.cmd).max(1) == 0
        if t % self.replan == 0:
            w = np.flatnonzero((self.mode != M_SKILL) & ~self.held)
            if len(w):
                wi = torch.as_tensor(w, device=self.device)
                cond = torch.as_tensor(joystick_condition(self.ck, self.cmd[w], self.clock0[w] + t), dtype=torch.float32, device=self.device)
                new = self.planner.plan(self.hist[wi, -self.C :], cond, self.gait[wi])
                if self.blend > 0:
                    k = torch.arange(1, self.blend + 1, device=self.device, dtype=torch.float32)[None, :, None] / (self.blend + 1)
                    new[:, : self.blend] = (1 - k) * self.queue[wi, : self.blend] + k * new[:, : self.blend]
                self.queue[wi, : self.H] = new
                self.queue[wi, self.H :] = new[:, -1:]
                self.seg[w] = np.where((self.mode[w] == M_STOP)[:, None], STOP, WALK)
                stop = self.mode[w] == M_STOP
                if stop.any():
                    ok = self._stood(new) & (np.abs(self.cmd[w]).max(1) == 0)
                    for j in np.flatnonzero(stop):
                        e = w[j]
                        self.ready[e] = self.ready[e] + 1 if ok[j] else 0
                        timeout = t - self.t_req[e] >= c.stop_timeout_s * FPS
                        if (self.ready[e] >= c.ready_replans and self._feet_down(e)) or timeout:
                            self._launch(e, timeout and self.ready[e] < c.ready_replans)
        goal = plan_goal_frames(self.queue[:, : self.K + 1], self.home, self.default_joint_pos)[:, : self.K].reshape(self.n, -1)
        self.seg_now = self.seg[:, 0].copy()
        self.mode_now = self.mode.copy()
        self.skill_now = np.where(self.mode == M_WALK, -1, self.skill)
        # advance: the goal's first frame is executed
        self.hist = torch.cat([self.hist[:, 1:], self.queue[:, :1]], 1)
        self.queue = torch.cat([self.queue[:, 1:], self.queue[:, -1:]], 1)
        self.seg = np.concatenate([self.seg[:, 1:], self.seg[:, -1:]], 1)
        sk = self.mode == M_SKILL
        self.remaining[sk] -= 1
        done = sk & (self.remaining <= 0)
        self.mode[done] = M_WALK
        self.skill[done] = -1
        self.t = t + 1
        return goal
