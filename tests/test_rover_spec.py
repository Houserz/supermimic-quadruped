"""Dobot Rover X1 port: spec dims, FK parity with MuJoCo, the mirror map, the motion library's
layout against this copy of the model, and an env smoke test on three library clips.

The library check is the import gate of plan decision E for this robot: a clip retargeted on a
model whose bodies are ordered or placed differently still loads and trains, scoring every link
against the wrong one."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.kinematics.robot_fk import RobotKinematics
from supermimic.robots import get_robot_spec

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "data" / "npz" / "rover"
CLIPS = ["ai4a_D1_055_KAN01_001_s14", "tencent_dog_quad_walk_002_s4", "cd2_hiFive_LFleg_posCtrl"]
needs_lib = pytest.mark.skipif(not all((LIB / f"{c}.npz").exists() for c in CLIPS), reason="library not built (scripts/data/build_dataset.py)")


def test_spec_dims():
    s = get_robot_spec("rover")
    assert s.nj == 12 and s.state_dim == 37 and s.delta_dim == 36 and s.obs_dim == 42 and s.goal_feat_dim == 34
    assert s.xi_dim == 16  # friction + 12 bias + com 3, no action delay
    m = s.load_model()
    assert m.nq == 19 and m.nv == 18 and m.nbody == 14
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]
    for b in s.tracked_bodies + (s.root_body, s.anchor_body, s.com_body):
        assert b in names, b
    geoms = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    import re

    assert sum(bool(re.match(s.foot_geom_regex, g)) for g in geoms) == 4


@pytest.mark.parametrize("dtype,atol_pos,atol_vel", [(torch.float64, 1e-10, 1e-9), (torch.float32, 2e-5, 2e-4)])
def test_rover_fk_matches_mujoco(dtype, atol_pos, atol_vel):
    model = get_robot_spec("rover").load_model()
    rng = np.random.default_rng(0)
    n = 64
    qpos = np.zeros((n, model.nq))
    qpos[:, :3] = rng.normal(size=(n, 3)) * 0.1
    q = rng.normal(size=(n, 4))
    qpos[:, 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    qpos[:, 7:] = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1])
    qvel = rng.normal(size=(n, model.nv)) * 2.0
    data = mujoco.MjData(model)
    pos_ref = np.zeros((n, model.nbody - 1, 3))
    lin_ref = np.zeros_like(pos_ref)
    ang_ref = np.zeros_like(pos_ref)
    res = np.zeros(6)
    for i in range(n):
        data.qpos[:], data.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_forward(model, data)
        pos_ref[i] = data.xpos[1:]
        for b in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
            ang_ref[i, b - 1], lin_ref[i, b - 1] = res[:3], res[3:]
    out = RobotKinematics(model, device="cpu", dtype=dtype)(torch.as_tensor(qpos, dtype=dtype), torch.as_tensor(qvel, dtype=dtype))
    np.testing.assert_allclose(out["pos"].numpy(), pos_ref, atol=atol_pos)
    np.testing.assert_allclose(out["ang_vel"].numpy(), ang_ref, atol=atol_vel)
    np.testing.assert_allclose(out["lin_vel"].numpy(), lin_ref, atol=atol_vel)


def test_mirror_map_matches_axes_home_and_fk():
    from supermimic.data.clips import MIRROR_POS, MIRROR_QUAT, body_states_from_pose
    from supermimic.robots.rover import HOME, JOINT_NAMES

    spec = get_robot_spec("rover")
    model = spec.load_model()
    axes = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_axis[i] for i in range(1, model.njnt)}
    for i, name in enumerate(JOINT_NAMES):
        partner = JOINT_NAMES[spec.mirror_joint_perm[i]]
        assert ("left" in name) != ("left" in partner)
        assert spec.mirror_joint_sign[i] == (1.0 if abs(axes[name][1]) > 0.5 else -1.0), name
        assert abs(spec.mirror_joint_sign[i] * HOME[partner] - HOME[name]) < 1e-12
    rng = np.random.default_rng(3)
    n = 8
    q = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1], size=(n, 12))
    rot = rng.normal(size=(n, 4))
    rot /= np.linalg.norm(rot, axis=-1, keepdims=True)
    pos = rng.normal(size=(n, 3)) * 0.1
    a = body_states_from_pose(model, spec, pos, rot, q)
    b = body_states_from_pose(model, spec, pos * MIRROR_POS, rot * MIRROR_QUAT, q[:, list(spec.mirror_joint_perm)] * np.array(spec.mirror_joint_sign))
    idx = [b.body_names.index(spec.mirror_body_map[nm]) for nm in a.body_names]
    # The nominal model is exactly symmetric, so the reflection is exact.
    assert np.abs(b.body_pos_w[:, idx] * MIRROR_POS - a.body_pos_w).max() < 1e-12
    # Control: without the abad sign flip the knees do not reflect.
    c = body_states_from_pose(model, spec, pos * MIRROR_POS, rot * MIRROR_QUAT, q[:, list(spec.mirror_joint_perm)])
    assert np.abs(c.body_pos_w[:, idx] * MIRROR_POS - a.body_pos_w).max() > 1e-2


@needs_lib
def test_library_layout_and_fk_match_this_model():
    from supermimic.data.clips import body_states_from_pose, load_clip, model_body_names

    spec = get_robot_spec("rover")
    model = spec.load_model()
    for name in CLIPS:
        c = load_clip(LIB / f"{name}.npz")
        assert c.joint_names == list(spec.joint_names) and c.body_names == model_body_names(model) and abs(c.fps - 50) < 1e-9
        ri = c.body_names.index(spec.root_body)
        rec = body_states_from_pose(model, spec, c.body_pos_w[:, ri], c.body_quat_w[:, ri], c.joint_pos, 1.0 / c.fps)
        assert np.abs(rec.body_pos_w - c.body_pos_w).max() < 1e-4
        assert np.abs(rec.body_lin_vel_w - c.body_lin_vel_w).max() < 1e-2


@needs_lib
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_env_smoke_on_library_clips():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env

    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(LIB / f"{c}.npz") for c in CLIPS], robot="rover", num_envs=16, replay_capacity=64, seed=0), device="cuda:0")
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (16, 420) and obs["goal"].shape == (16, 340)
        assert env.action_manager.total_action_dim == 12
        for _ in range(20):
            obs, rew, term, tout, _ = env.step(0.3 * torch.randn(16, 12, device=env.device))
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all()
        rec = env.recorder_manager.get_term("replay")
        assert rec.ctx.xi_dim == 16 and rec.buffer.s.shape[-1] == 37
    finally:
        env.close()


def test_stiffness_override_reaches_the_actuators_and_keeps_the_damping_ratio():
    """The feasibility screen runs at kp 120; training must still get the robot's kp 25 / kd 1.3."""
    from supermimic.robots import set_robot_overrides
    from supermimic.robots.rover import DAMPING, STIFFNESS

    try:
        set_robot_overrides()
        acts = get_robot_spec("rover").entity_cfg_fn(False).articulation.actuators
        assert all(a.stiffness == STIFFNESS == 25.0 and a.damping == DAMPING == 1.3 for a in acts)
        assert sorted(a.effort_limit for a in acts) == [23.0, 55.0]
        set_robot_overrides(stiffness=100.0)
        acts = get_robot_spec("rover").entity_cfg_fn(False).articulation.actuators
        assert all(a.stiffness == 100.0 and a.damping == pytest.approx(2.6) for a in acts)  # kd x sqrt(4)
        set_robot_overrides(action_scale=0.5)
        with pytest.raises(ValueError):
            get_robot_spec("rover")
    finally:
        set_robot_overrides()
    assert get_robot_spec("rover").entity_cfg_fn(False).articulation.actuators[0].stiffness == 25.0
