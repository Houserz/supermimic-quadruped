"""Clip utilities (numpy + CPU MuJoCo): FK re-computation for the import check, left-right
mirroring and static-frame trimming, for any robot spec.

Velocities by ``mj_differentiatePos`` between consecutive 50 Hz frames, link states by
``mj_objectVelocity`` on ``mjOBJ_BODY`` -- the same routine as ``quadmotion.clip`` -- so a clip
exported there and re-derived here from its joint angles and root pose agrees to numerical
precision; any disagreement means a model or ordering mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from supermimic.robots.spec import RobotSpec

CONTROL_DT = 0.02


@dataclass
class ClipArrays:
    joint_pos: np.ndarray  # (T, nj)
    joint_vel: np.ndarray
    body_pos_w: np.ndarray  # (T, B, 3)
    body_quat_w: np.ndarray
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


def model_body_names(model: mujoco.MjModel) -> list[str]:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    assert names[0] == "world"
    return names[1:]


def body_states_from_pose(model: mujoco.MjModel, spec: RobotSpec, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray, dt: float = CONTROL_DT) -> ClipArrays:
    """FK + finite-difference velocities for every entity body."""
    n = len(joint_pos)
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm) for nm in spec.joint_names]
    q_adr = np.array([model.jnt_qposadr[j] for j in jid])
    v_adr = np.array([model.jnt_dofadr[j] for j in jid])
    data = mujoco.MjData(model)
    names = model_body_names(model)
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
    return ClipArrays(joint_pos.copy(), joint_vel, body_pos, body_quat, body_lin, body_ang, list(spec.joint_names), names, 1.0 / dt)


def quat_signed_err(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    sign = np.sign(np.sum(q * q_ref, axis=-1, keepdims=True))
    sign[sign == 0] = 1.0
    return np.abs(q * sign - q_ref).max(-1)


# ---------------------------------------------------------------- mirroring (plan 2.6)

MIRROR_POS = np.array([1.0, -1.0, 1.0])
MIRROR_QUAT = np.array([1.0, -1.0, 1.0, -1.0])  # reflection through the x-z plane: (w,x,y,z) -> (w,-x,y,-z)


def mirror_clip(clip: ClipArrays, spec: RobotSpec, model: mujoco.MjModel) -> ClipArrays:
    """Left-right mirror: joints permuted / sign-flipped by the spec's mirror map, root pose
    reflected, and **every body state re-derived by FK** (body frames of mirror links are not
    exact mirror images in the MJCF, so mirroring link poses directly would be wrong)."""
    perm = np.array(spec.mirror_joint_perm)
    sign = np.array(spec.mirror_joint_sign)
    q_m = clip.joint_pos[:, perm] * sign
    root_i = clip.body_names.index(spec.root_body)
    root_pos = clip.body_pos_w[:, root_i] * MIRROR_POS
    root_quat = clip.body_quat_w[:, root_i] * MIRROR_QUAT
    return body_states_from_pose(model, spec, root_pos, root_quat, q_m, 1.0 / clip.fps)


def mirror_consistency(clip: ClipArrays, mirrored: ClipArrays, spec: RobotSpec) -> float:
    """Max |mirror(original body position) - mirrored body position| over the tracked bodies (m);
    checks the joint sign convention: a wrong sign shows up as centimetres, not microns."""
    bm = spec.mirror_body_map
    err = 0.0
    for name in spec.tracked_bodies:
        i = clip.body_names.index(name)
        j = mirrored.body_names.index(bm[name])
        err = max(err, float(np.abs(clip.body_pos_w[:, i] * MIRROR_POS - mirrored.body_pos_w[:, j]).max()))
    return err


def is_symmetric(clip: ClipArrays, mirrored: ClipArrays, tol_rad: float = 0.02) -> bool:
    """True when the mirrored clip is the clip itself (RMS joint difference below ``tol_rad``)."""
    return float(np.sqrt(np.mean((clip.joint_pos - mirrored.joint_pos) ** 2))) < tol_rad


# ---------------------------------------------------------------- static trimming (plan 2.6)


def static_mask(joint_vel: np.ndarray, speed_threshold: float = 0.05) -> np.ndarray:
    """Frames whose mean |joint velocity| is below ``speed_threshold`` rad/s."""
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
    start = max(0, lead - keep)
    end = n - max(0, trail - keep)
    return start, end


def slice_clip(clip: ClipArrays, start: int, end: int) -> ClipArrays:
    return ClipArrays(
        clip.joint_pos[start:end], clip.joint_vel[start:end], clip.body_pos_w[start:end], clip.body_quat_w[start:end],
        clip.body_lin_vel_w[start:end], clip.body_ang_vel_w[start:end], clip.joint_names, clip.body_names, clip.fps,
    )
