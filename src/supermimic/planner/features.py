"""Planner frame features and their inverse (numpy).

A frame is the tracker's goal quantities (``motion_library.goal_features``) minus the joint
velocities, in a form that integrates back to a pose:

    joint_pos - HOME (nj) | root height (1) | gravity in the root frame (3)
    | root linear velocity in the heading frame (3) | world yaw rate (1)

Nothing in it is global: position and yaw are recovered by integrating the velocities from a
chosen start, which is all the tracker needs (its goal features are local too).
"""

from __future__ import annotations

import numpy as np

FPS = 50.0


def feat_dim(nj: int) -> int:
    return nj + 8


def feat_slices(nj: int) -> dict[str, slice]:
    return {"joint": slice(0, nj), "height": slice(nj, nj + 1), "grav": slice(nj + 1, nj + 4), "lin_h": slice(nj + 4, nj + 7), "yaw_rate": slice(nj + 7, nj + 8)}


def yaw_of(quat: np.ndarray) -> np.ndarray:
    """ZYX yaw of wxyz quaternions (..., 4)."""
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_rotate_inverse(quat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vectors v (..., 3) into the frame of wxyz quaternions (..., 4)."""
    w = quat[..., :1]
    u = -quat[..., 1:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_from_zyx(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], axis=-1)


def features_from_pose(root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray, home: np.ndarray, fps: float = FPS) -> np.ndarray:
    """(T, nj+8) features of a clip; velocities by central differences (one-sided at the ends)."""
    dt = 1.0 / fps
    yaw = np.unwrap(yaw_of(root_quat))
    vel_w = np.gradient(root_pos, dt, axis=0)
    c, s = np.cos(yaw), np.sin(yaw)
    lin_h = np.stack([c * vel_w[:, 0] + s * vel_w[:, 1], -s * vel_w[:, 0] + c * vel_w[:, 1], vel_w[:, 2]], axis=-1)
    yaw_rate = np.gradient(yaw, dt)
    grav = quat_rotate_inverse(root_quat, np.broadcast_to(np.array([0.0, 0.0, -1.0]), root_pos.shape))
    return np.concatenate([joint_pos - home, root_pos[:, 2:3], grav, lin_h, yaw_rate[:, None]], axis=-1)


def pose_from_features(feat: np.ndarray, home: np.ndarray, fps: float = FPS, xy0=(0.0, 0.0), yaw0: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate features back to (root_pos (T,3), root_quat (T,4) wxyz, joint_pos (T,nj)).

    Yaw and xy are integrated with the trapezoidal rule (the inverse of central differences up to
    O(dt^2)); roll / pitch come from the gravity direction (ZYX: g_b = (sin p, -sin r cos p, -cos r cos p)).
    """
    nj = len(home)
    sl = feat_slices(nj)
    dt = 1.0 / fps
    wz = feat[:, sl["yaw_rate"]][:, 0]
    yaw = yaw0 + np.concatenate([[0.0], np.cumsum(0.5 * (wz[1:] + wz[:-1]) * dt)])
    lin_h = feat[:, sl["lin_h"]]
    c, s = np.cos(yaw), np.sin(yaw)
    vx_w = c * lin_h[:, 0] - s * lin_h[:, 1]
    vy_w = s * lin_h[:, 0] + c * lin_h[:, 1]
    x = xy0[0] + np.concatenate([[0.0], np.cumsum(0.5 * (vx_w[1:] + vx_w[:-1]) * dt)])
    y = xy0[1] + np.concatenate([[0.0], np.cumsum(0.5 * (vy_w[1:] + vy_w[:-1]) * dt)])
    g = feat[:, sl["grav"]]
    g = g / np.linalg.norm(g, axis=-1, keepdims=True)
    pitch = np.arcsin(np.clip(g[:, 0], -1.0, 1.0))
    roll = np.arctan2(-g[:, 1], -g[:, 2])
    root_pos = np.stack([x, y, feat[:, sl["height"]][:, 0]], axis=-1)
    root_quat = quat_from_zyx(yaw, pitch, roll)
    joint_pos = feat[:, sl["joint"]] + home
    return root_pos, root_quat, joint_pos


def heading_velocity(root_pos: np.ndarray, root_quat: np.ndarray, fps: float = FPS, window: int = 25) -> np.ndarray:
    """(T, 3) centered ``window``-frame mean of (vx, vy) in the heading frame and the yaw rate.

    The planner's condition, as the paper's velocity interface defines a reference command (the
    centered 25-frame mean of the reference root velocity in its heading frame). Edges use the
    available frames only.
    """
    f = features_from_pose(root_pos, root_quat, np.zeros((len(root_pos), 0)), np.zeros(0), fps)
    v = np.concatenate([f[:, 4:6], f[:, 7:8]], axis=-1)  # lin_h xy, yaw rate (nj = 0)
    k = np.ones(window)
    num = np.stack([np.convolve(v[:, j], k, mode="same") for j in range(3)], axis=-1)
    den = np.convolve(np.ones(len(v)), k, mode="same")[:, None]
    return num / den
