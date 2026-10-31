"""The two places where a sign or a convention would make a plausible but wrong clip.

* CyberDog's ``RBodyMat`` order and the inverse body motion: get either wrong and "sit down"
  raises the rear instead of lowering it, or the planted feet wander — both still produce
  smooth, IK-feasible clips.
* Froude time scaling: without it a scaled jump needs a different gravity, which no check on
  kinematics alone notices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quadmotion import cyberdog, robot
from quadmotion.clip import FPS, froude_time_factor, resample

ROOT = Path(__file__).resolve().parents[1]
CD2 = ROOT / "data" / "raw" / "cyberdog2" / "motion_list" / "preinstalled"
needs_raw = pytest.mark.skipif(not CD2.exists(), reason="run scripts/data/fetch_sources.py first")


def _feet_world(m: cyberdog.CyberdogMotion) -> np.ndarray:
    R = Rotation.from_quat(m.root_quat, scalar_first=True).as_matrix()
    return m.root_pos[:, None] + np.einsum("tij,tlj->tli", R, m.feet_b)


@needs_raw
def test_sit_down_lowers_the_rear_and_keeps_the_feet():
    rover = robot.load_robot("rover")
    m = cyberdog.interpret(CD2 / "cyberdog2_sitDown_posCtrl.toml", rover)
    R = Rotation.from_quat(m.root_quat, scalar_first=True).as_matrix()
    front = m.root_pos + R @ np.array([rover.hip_x, 0, 0])
    rear = m.root_pos + R @ np.array([-rover.hip_x, 0, 0])
    k = int(np.argmax(front[:, 2] - rear[:, 2]))
    assert front[k, 2] - rear[k, 2] > 0.15  # nose up, rear down
    fw = _feet_world(m)
    # 1e-6 m: the 50 Hz resampling (slerp of the trunk, linear feet) leaves ~1e-7 m.
    assert m.slide < 1e-9 and np.abs(fw[:, :, :2] - fw[0, :, :2]).max() < 1e-6
    assert np.abs(fw[:, :, 2] - rover.foot_radius).max() < 1e-6


def test_rbody_mat_is_z_then_y_then_x():
    """Control for the sign above: a pure pitch command must rotate about y only."""
    M = cyberdog.rbody_mat(0.0, 0.3, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(M[:3, :3], Rotation.from_euler("y", 0.3).as_matrix())
    M = cyberdog.rbody_mat(0.2, 0.3, 0.1, 0, 0, 0)
    assert np.allclose(M[:3, :3], Rotation.from_euler("z", 0.2).as_matrix() @ Rotation.from_euler("y", 0.3).as_matrix() @ Rotation.from_euler("x", 0.1).as_matrix())


@needs_raw
def test_high_five_lifts_only_the_named_paw():
    rover = robot.load_robot("rover")
    m = cyberdog.interpret(CD2 / "cyberdog2_hiFive_LFleg_posCtrl.toml", rover)
    lift = _feet_world(m)[:, :, 2].max(0) - rover.foot_radius
    assert lift[robot.LEGS.index("FL")] > 0.3
    assert np.all(np.delete(lift, robot.LEGS.index("FL")) < 1e-6)


def test_froude_scaling_keeps_a_ballistic_arc_ballistic():
    g = 9.81
    t = np.arange(0.0, 0.4, 1.0 / FPS)
    z = 1.5 * t - 0.5 * g * t**2  # dog-scale flight
    s = 0.7
    pos = np.stack([np.zeros_like(t), np.zeros_like(t), s * z], -1)
    quat = np.tile([1.0, 0, 0, 0], (len(t), 1))
    tf = froude_time_factor(s)
    t_out = np.arange(0.0, t[-1] * tf, 1.0 / FPS) / tf
    zr = resample(t, t_out, pos, quat)[0][:, 2]
    acc = np.diff(zr, 2) * FPS**2
    assert abs(np.median(acc) + g) < 0.05 * g
    # Control: scaling space alone gives s * g.
    acc_bad = np.diff(s * z, 2) * FPS**2
    assert abs(np.median(acc_bad) + s * g) < 0.05 * g
