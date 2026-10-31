"""Clips in the mjlab motion layout, the one ``supermimic`` trains on.

``ClipArrays`` / ``to_npz_dict`` / ``body_states_from_pose`` / ``trim_bounds`` match
``supermimic.data.clips`` (velocities by ``mj_differentiatePos`` between consecutive 50 Hz frames,
link states by ``mj_objectVelocity`` on ``mjOBJ_BODY``), so the training side's import check
(FK re-computation, 1e-4 m) holds for every exported file. Joint arrays follow the model's order.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from quadmotion.robot import Quadruped, body_names, model_joint_names

FPS = 50.0
CONTROL_DT = 1.0 / FPS


@dataclass
class ClipArrays:
    joint_pos: np.ndarray  # (T, 12)
    joint_vel: np.ndarray
    body_pos_w: np.ndarray  # (T, B, 3)
    body_quat_w: np.ndarray  # (T, B, 4) wxyz
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    joint_names: list[str]
    body_names: list[str]
    fps: float

    @property
    def frames(self) -> int:
        return int(self.joint_pos.shape[0])

    def to_npz_dict(self) -> dict:
        return dict(
            joint_pos=self.joint_pos.astype(np.float32),
            joint_vel=self.joint_vel.astype(np.float32),
            body_pos_w=self.body_pos_w.astype(np.float32),
            body_quat_w=self.body_quat_w.astype(np.float32),
            body_lin_vel_w=self.body_lin_vel_w.astype(np.float32),
            body_ang_vel_w=self.body_ang_vel_w.astype(np.float32),
            joint_names=np.array(self.joint_names),
            body_names=np.array(self.body_names),
            fps=np.array(self.fps),
        )


def load_clip(path) -> ClipArrays:
    with np.load(path) as d:
        return ClipArrays(
            joint_pos=np.asarray(d["joint_pos"], np.float64),
            joint_vel=np.asarray(d["joint_vel"], np.float64),
            body_pos_w=np.asarray(d["body_pos_w"], np.float64),
            body_quat_w=np.asarray(d["body_quat_w"], np.float64),
            body_lin_vel_w=np.asarray(d["body_lin_vel_w"], np.float64),
            body_ang_vel_w=np.asarray(d["body_ang_vel_w"], np.float64),
            joint_names=[str(n) for n in d["joint_names"]],
            body_names=[str(n) for n in d["body_names"]],
            fps=float(np.asarray(d["fps"]).reshape(-1)[0]),
        )


def body_states_from_pose(model: mujoco.MjModel, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray, dt: float = CONTROL_DT) -> ClipArrays:
    """FK + finite-difference velocities for every body (root pose wxyz, joints in model order)."""
    n = len(joint_pos)
    joint_names = model_joint_names(model)
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm) for nm in joint_names]
    q_adr = np.array([model.jnt_qposadr[j] for j in jid])
    v_adr = np.array([model.jnt_dofadr[j] for j in jid])
    data = mujoco.MjData(model)
    names = body_names(model)
    nb = len(names)
    body_pos = np.zeros((n, nb, 3))
    body_quat = np.zeros((n, nb, 4))
    body_lin = np.zeros((n, nb, 3))
    body_ang = np.zeros((n, nb, 3))
    joint_vel = np.zeros((n, len(jid)))

    def set_qpos(k):
        data.qpos[:] = 0.0
        data.qpos[0:3] = root_pos[k]
        data.qpos[3:7] = root_quat[k]
        data.qpos[q_adr] = joint_pos[k]

    qa, qb, qvel, vel6 = np.zeros(model.nq), np.zeros(model.nq), np.zeros(model.nv), np.zeros(6)
    for k in range(n):
        nxt = min(k + 1, n - 1)
        set_qpos(k)
        qa[:] = data.qpos
        set_qpos(nxt)
        qb[:] = data.qpos
        if nxt != k:
            mujoco.mj_differentiatePos(model, qvel, dt, qa, qb)
        else:
            qvel[:] = 0.0
        set_qpos(k)
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        joint_vel[k] = qvel[v_adr]
        body_pos[k] = data.xpos[1:]
        body_quat[k] = data.xquat[1:]
        for b in range(nb):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b + 1, vel6, 0)
            body_ang[k, b] = vel6[0:3]
            body_lin[k, b] = vel6[3:6]
    return ClipArrays(joint_pos.copy(), joint_vel, body_pos, body_quat, body_lin, body_ang, list(joint_names), names, 1.0 / dt)


# ---------------------------------------------------------------- time scaling


def froude_time_factor(scale: float) -> float:
    """Durations scale with sqrt(length) for dynamic similarity under the same gravity: a
    ballistic arc shrunk by ``s`` in space stays ballistic only if it is played ``sqrt(s)`` times
    as fast. Without it a 0.7x dog jump would need 0.7 g in flight."""
    return float(np.sqrt(scale))


def resample(t_src: np.ndarray, t_out: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray, linear: dict[str, np.ndarray] | None = None, nearest: dict[str, np.ndarray] | None = None):
    """Root position and other arrays linearly, root orientation by slerp, flags by nearest sample."""
    from scipy.spatial.transform import Rotation, Slerp

    t_out = np.clip(t_out, t_src[0], t_src[-1])

    def lin(a: np.ndarray) -> np.ndarray:
        flat = a.reshape(len(a), -1)
        return np.stack([np.interp(t_out, t_src, flat[:, j]) for j in range(flat.shape[1])], -1).reshape((len(t_out),) + a.shape[1:])

    quat = Slerp(t_src, Rotation.from_quat(root_quat, scalar_first=True))(t_out).as_quat(scalar_first=True)
    idx = np.clip(np.searchsorted(t_src, t_out), 1, len(t_src) - 1)
    idx = np.where(np.abs(t_src[idx - 1] - t_out) <= np.abs(t_src[idx] - t_out), idx - 1, idx)
    return (
        lin(root_pos),
        quat,
        {k: lin(v) for k, v in (linear or {}).items()},
        {k: v[idx] for k, v in (nearest or {}).items()},
    )


# ---------------------------------------------------------------- mirroring

MIRROR_POS = np.array([1.0, -1.0, 1.0])
MIRROR_QUAT = np.array([1.0, -1.0, 1.0, -1.0])  # reflection through the x-z plane: (w,x,y,z) -> (w,-x,y,-z)


def mirror_pose(robot: Quadruped, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return root_pos * MIRROR_POS, root_quat * MIRROR_QUAT, joint_pos[:, list(robot.mirror_perm)] * robot.mirror_sign


def mirror_clip(clip: ClipArrays, robot: Quadruped) -> ClipArrays:
    """Left-right mirror; every body state is re-derived by FK rather than reflected link by link."""
    ri = clip.body_names.index(robot.root_body)
    rp, rq, q = mirror_pose(robot, clip.body_pos_w[:, ri], clip.body_quat_w[:, ri], clip.joint_pos)
    return body_states_from_pose(robot.model, rp, rq, q, 1.0 / clip.fps)


# ---------------------------------------------------------------- trimming


def static_mask(joint_vel: np.ndarray, speed_threshold: float = 0.05) -> np.ndarray:
    return np.abs(joint_vel).mean(-1) < speed_threshold


def trim_bounds(joint_vel: np.ndarray, fps: float, keep_s: float = 1.0, speed_threshold: float = 0.05) -> tuple[int, int]:
    """[start, end) keeping at most ``keep_s`` seconds of leading / trailing static frames."""
    n = len(joint_vel)
    st = static_mask(joint_vel, speed_threshold)
    lead = 0
    while lead < n and st[lead]:
        lead += 1
    trail = 0
    while trail < n - lead and st[n - 1 - trail]:
        trail += 1
    keep = int(round(keep_s * fps))
    return max(0, lead - keep), n - max(0, trail - keep)


def slice_clip(clip: ClipArrays, start: int, end: int) -> ClipArrays:
    return ClipArrays(
        clip.joint_pos[start:end], clip.joint_vel[start:end], clip.body_pos_w[start:end], clip.body_quat_w[start:end],
        clip.body_lin_vel_w[start:end], clip.body_ang_vel_w[start:end], clip.joint_names, clip.body_names, clip.fps,
    )
