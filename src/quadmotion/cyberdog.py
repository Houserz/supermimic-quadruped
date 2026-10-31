"""Xiaomi CyberDog2 motion lists (``motion_list/*.toml``) replayed on a 12-DoF quadruped robot's geometry.

Only the kinematic step types are interpreted; a file using any other type is reported and
skipped (``UnsupportedStep``):

``pose`` / ``swingleg``  (``fsm_state_posectrl.cpp`` + ``PositionController::TransBody``)
    Every 2 ms tick, each *support* foot's position relative to the trunk centre ``p`` becomes
    ``M (p + c) - c`` with ``M = RBodyMat(yaw, pitch, roll, -x, -y, -z)`` built from the step's
    ``body_cmd / ticks`` and ``c`` the ``ctrl_point``; a *swing* foot moves by ``foot_pose / ticks``
    in the trunk frame. Support feet stay put in the world, which fixes the trunk's world motion:
    ``T_world_trunk <- T_world_trunk @ G^-1`` with ``G`` the affine map above.
``transition``  (gait_id 5): stand at ``height`` (the first one also sets the initial stance, feet
    under the hips). CyberDog interpolates joint angles, which drags planted feet when they are
    not under the hips; here the feet stay where they are in the world and only the trunk is
    levelled over them, so the reference never asks for a sliding foot.
``recoverystand``: the same at 0.24 m (``fsm_state_recoverystand.cpp``: "cyberdog2: 0.24").

CyberDog2 leg order is FR, FL, RR, RL (``PositionController::IKS`` side signs -1, 1, -1, 1).
Lengths are scaled to the robot by the ratio of leg lengths (thigh + calf: Rover X1 0.46 m vs 0.294 m,
``cyberdog2-description.yaml``); angles are not scaled. Step durations are in ms and the
interpreter's ``duration / 2`` ticks of 2 ms reproduce them; the result is then played
``sqrt(s)`` = 1.25 times as slow (``clip.froude_time_factor``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from quadmotion.clip import FPS, froude_time_factor, resample
from quadmotion.robot import LEGS, Quadruped

CD2_THIGH, CD2_CALF = 0.12, 0.17391  # cyberdog2-description.yaml hip_link_length, knee_link_length
CD2_LEG_ORDER = ("FR", "FL", "RR", "RL")
RECOVERY_HEIGHT = 0.24
TICK = 0.002
KINEMATIC = {"pose", "swingleg", "transition", "recoverystand"}


class UnsupportedStep(Exception):
    pass


@dataclass
class CyberdogMotion:
    name: str
    root_pos: np.ndarray  # (T, 3) at 50 Hz
    root_quat: np.ndarray  # (T, 4) wxyz
    feet_b: np.ndarray  # (T, 4, 3) foot centres in the trunk frame, LEGS order
    support: np.ndarray  # (T, 4) bool, LEGS order
    slide: float  # max horizontal motion of a support foot over the clip (m); transitions can slide


def scale_for(robot: Quadruped) -> float:
    g = robot.legs["FL"]
    return (g.l_thigh + g.l_calf) / (CD2_THIGH + CD2_CALF)


def rbody_mat(alpha: float, beta: float, gamma: float, x: float, y: float, z: float) -> np.ndarray:
    """``PositionController::RBodyMat``: Rz(alpha) Ry(beta) Rx(gamma) with translation (x, y, z)."""
    M = np.eye(4)
    M[:3, :3] = Rotation.from_euler("ZYX", [alpha, beta, gamma]).as_matrix()
    M[:3, 3] = (x, y, z)
    return M


def _translate(c: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = c
    return T


def interpret(path: Path, robot: Quadruped, amplitude: float = 1.0, time_scaling: bool = True, after: Path | None = None) -> CyberdogMotion:
    """``amplitude`` scales every pose step's ``body_cmd`` (trunk rotation and translation), not
    the swing foot's ``foot_pose``. The Rover X1's legs are longer relative to its 0.7 m trunk box
    than CyberDog2's, so a deep sit can put a rear knee into the ground; the build lowers the
    amplitude until every frame passes the checks and records the value it used.

    ``after`` is a motion list played first. Some files continue from the state another one
    leaves the robot in (``sitDown_posCtrl_left`` opens with ``x +0.08``, undoing the last step
    of ``sitDown_posCtrl``); played from a stand they would be a different motion."""
    steps = list(tomllib.loads(Path(after).read_text())["step"]) if after else []
    steps += tomllib.loads(Path(path).read_text())["step"]
    bad = sorted({s["type"] for s in steps} - KINEMATIC)
    if bad:
        raise UnsupportedStep(f"needs the CyberDog controller for step types {bad}")
    s = scale_for(robot)
    r = robot.foot_radius
    thigh = np.stack([robot.legs[l].thigh_pos for l in LEGS])
    cd_index = [CD2_LEG_ORDER.index(l) for l in LEGS]

    def stance(height: float) -> np.ndarray:
        p = thigh.copy()
        p[:, 2] = -(s * height - r)
        return p

    first = steps[0]
    h0 = first.get("height", RECOVERY_HEIGHT) if first["type"] == "transition" else RECOVERY_HEIGHT
    T_wt = np.eye(4)
    T_wt[2, 3] = s * h0
    feet = stance(h0)  # trunk frame
    support = np.ones(4, bool)
    samples_T, samples_feet, samples_sup = [], [], []
    world_support_ref: dict[int, np.ndarray] = {}
    slide = 0.0

    def record():
        nonlocal slide
        samples_T.append(T_wt.copy())
        samples_feet.append(feet.copy())
        samples_sup.append(support.copy())
        fw = (T_wt[:3, :3] @ feet.T).T + T_wt[:3, 3]
        for i in range(4):
            if support[i]:
                ref = world_support_ref.setdefault(i, fw[i].copy())
                slide = max(slide, float(np.linalg.norm(fw[i, :2] - ref[:2])))
            else:
                world_support_ref.pop(i, None)

    record()
    for st in steps:
        n = max(1, int(st.get("duration", 0)) // 2)
        kind = st["type"]
        if kind in ("pose", "swingleg"):
            fs = st.get("foot_support", [1, 1, 1, 1])
            sup = np.array([fs[cd_index[i]] > 0 for i in range(4)])
            if not sup.any():
                sup[:] = True  # contact == 0 is treated as all four (fsm_state_posectrl.cpp)
            bc = np.asarray(st.get("body_cmd", [0.0] * 6), float) * amplitude
            fp = np.asarray(st.get("foot_pose", [0.0] * 3), float)[:3] * s
            c = np.asarray(st.get("ctrl_point", [0.0] * 3), float)[:3] * s
            M = rbody_mat(bc[2] / n, bc[1] / n, bc[0] / n, -s * bc[3] / n, -s * bc[4] / n, -s * bc[5] / n)
            G = _translate(-c) @ M @ _translate(c)
            G_inv = np.linalg.inv(G)
            support = sup
            for _ in range(n):
                feet_h = np.c_[feet, np.ones(4)]
                moved = (G @ feet_h.T).T[:, :3]
                feet = np.where(sup[:, None], moved, feet + fp / n)
                T_wt = T_wt @ G_inv
                record()
        else:  # transition / recoverystand: level the trunk over the feet at the target height
            height = st.get("height", RECOVERY_HEIGHT) if kind == "transition" else RECOVERY_HEIGHT
            height = float(np.clip(height, 0.10, 0.30))  # fsm_state_posectrl.cpp clamps the same way
            fw = (T_wt[:3, :3] @ feet.T).T + T_wt[:3, 3]
            yaw = Rotation.from_matrix(T_wt[:3, :3]).as_euler("ZYX")[0]
            R_end = Rotation.from_euler("Z", yaw).as_matrix()
            centre = fw[:, :2].mean(0) - (R_end[:2, :2] @ thigh[:, :2].mean(0))
            T_end = np.eye(4)
            T_end[:3, :3] = R_end
            T_end[:3, 3] = (*centre, s * height)
            fw_end = fw.copy()
            fw_end[:, 2] = r
            rot0 = Rotation.from_matrix(T_wt[:3, :3])
            p0 = T_wt[:3, 3].copy()
            key = Rotation.concatenate([rot0, Rotation.from_matrix(R_end)])
            support = np.ones(4, bool)
            for k in range(1, n + 1):
                a = 0.5 - 0.5 * np.cos(np.pi * k / n)  # smooth ramp
                T_wt = np.eye(4)
                T_wt[:3, :3] = (key[0] * Rotation.from_rotvec(a * (key[0].inv() * key[1]).as_rotvec())).as_matrix()
                T_wt[:3, 3] = (1 - a) * p0 + a * T_end[:3, 3]
                fwk = (1 - a) * fw + a * fw_end
                feet = (T_wt[:3, :3].T @ (fwk - T_wt[:3, 3]).T).T
                record()
    Ts = np.stack(samples_T)
    # Robot time: CyberDog's 2 ms ticks stretched by sqrt(s) (clip.froude_time_factor), at 50 Hz.
    tf = froude_time_factor(s) if time_scaling else 1.0
    t_src = np.arange(len(Ts)) * TICK * tf
    t_out = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / FPS)
    root_pos, root_quat, lin, near = resample(
        t_src, t_out, Ts[:, :3, 3], Rotation.from_matrix(Ts[:, :3, :3]).as_quat(scalar_first=True),
        {"feet": np.stack(samples_feet)}, {"support": np.stack(samples_sup)},
    )
    return CyberdogMotion(Path(path).stem, root_pos, root_quat, lin["feet"], near["support"], slide)
