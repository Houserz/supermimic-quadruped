"""Rover walking planner: the shared pieces the recording, the closed loop and the teleop rely on.

Silent failures pinned here:

* the 20-dim Rover frame (12 joints) does not integrate back to the pose it came from -- a plan
  would then walk somewhere else than its features say, and FK / rendering / the tracker's goal
  would all inherit it; the control drops the yaw rate and must miss by metres;
* a plan's tracker goal (``planner.goal.plan_goal_frames``) is not the goal the motion library
  computes for the same motion -- the tracker would be fed a goal laid out or scaled unlike the
  one it was trained on, and would still run; the control shifts the plan by one frame and must
  exceed the tolerance, so the comparison resolves a frame;
* the command sampler holds a command inside the velocity policy's standing switch
  (0 < |c| < 0.1), where the recorded robot stands while the planner's condition says walk; the
  control turns the rule off and must produce such holds, so the check is not vacuous;
* the foot is put somewhere other than the foot sphere (the Rover has no foot bodies).
* the clock-conditioned planner gets a clock other than the walking policy's, or a checkpoint
  of it is rebuilt with the base model's 3-wide condition.
* actual-velocity planners: the recording's ramps exceed their rate cap; the clock is gated on
  the joystick value itself instead of the policy command that walks at it (H(v)), or a
  command-conditioned checkpoint gets a condition other than its own.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from supermimic.data.clips import body_states_from_pose
from supermimic.envs.mdp.motion_library import RefState, goal_features
from supermimic.planner.features import features_from_pose, pose_from_features, quat_from_zyx
from supermimic.planner.goal import plan_goal_frames
from supermimic.planner.rover import (
    FEAT_DIM,
    FOOT_GEOMS,
    HOME_VEC,
    NJ,
    PHASE_THRESHOLD,
    CommandCfg,
    RoverCommands,
    foot_centres,
    foot_offsets,
    phase_condition,
)
from supermimic.robots import get_robot_spec

FPS = 50.0


def synthetic_walk(seconds: float = 10.0, oversample: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A smooth turning walk: heading-frame velocity, yaw, roll / pitch and joint motion, with the root
    path integrated on a grid ``oversample`` times finer than 50 Hz and then sampled at 50 Hz."""
    n = int(seconds * FPS)
    tf = np.arange(n * oversample) / (FPS * oversample)
    yaw = 0.8 * tf + 0.2 * np.sin(tf)
    vx, vy = 0.5 + 0.1 * np.sin(2 * tf), 0.1 * np.cos(1.3 * tf)
    vw = np.stack([np.cos(yaw) * vx - np.sin(yaw) * vy, np.sin(yaw) * vx + np.cos(yaw) * vy], 1)
    xy = np.cumsum(vw, 0) / (FPS * oversample)
    t = tf[::oversample]
    root_pos = np.concatenate([xy[::oversample], (0.39 + 0.01 * np.sin(4 * t))[:, None]], 1)
    root_quat = quat_from_zyx(yaw[::oversample], 0.04 * np.cos(2.5 * t), 0.05 * np.sin(3 * t))
    phase = np.arange(NJ) * 0.7
    q = HOME_VEC + 0.3 * np.sin(2 * np.pi * 1.6 * t[:, None] + phase)
    return root_pos, root_quat, q


def test_feature_round_trip_rover():
    rp, rq, q = synthetic_walk()
    feat = features_from_pose(rp, rq, q, HOME_VEC)
    assert feat.shape == (len(q), FEAT_DIM) and FEAT_DIM == 20
    # The walk must exercise the integration: metres travelled and radians turned.
    assert np.linalg.norm(rp[-1, :2] - rp[0, :2]) > 1.0 and abs(feat[:, -1].sum() / FPS) > 5.0
    p2, r2, q2 = pose_from_features(feat, HOME_VEC, xy0=rp[0, :2], yaw0=float(np.arctan2(2 * (rq[0, 0] * rq[0, 3] + rq[0, 1] * rq[0, 2]), 1 - 2 * (rq[0, 2] ** 2 + rq[0, 3] ** 2))))
    np.testing.assert_allclose(q2, q, atol=1e-12)
    np.testing.assert_allclose(p2[:, 2], rp[:, 2], atol=1e-12)
    assert np.abs(p2[:, :2] - rp[:, :2]).max() < 5e-3  # 10 s, 9 m of turning walk: central difference + trapezoid
    assert (1 - np.abs((r2 * rq).sum(-1))).max() < 1e-5
    np.testing.assert_allclose(features_from_pose(p2, r2, q2, HOME_VEC)[2:-2], feat[2:-2], atol=2e-3)  # one-sided differences at the ends
    # Control: without the yaw rate the same velocities integrate along a straight line.
    bad = feat.copy()
    bad[:, -1] = 0.0
    p3, _, _ = pose_from_features(bad, HOME_VEC, xy0=rp[0, :2])
    assert np.abs(p3[:, :2] - rp[:, :2]).max() > 1.0


def _clip_goal(clip) -> torch.Tensor:
    ref = RefState(root_pos_w=torch.as_tensor(clip.body_pos_w[:, 0]), root_quat_w=torch.as_tensor(clip.body_quat_w[:, 0]), joint_pos=torch.as_tensor(clip.joint_pos),
                   root_lin_vel_w=torch.as_tensor(clip.body_lin_vel_w[:, 0]), root_ang_vel_w=torch.as_tensor(clip.body_ang_vel_w[:, 0]), joint_vel=torch.as_tensor(clip.joint_vel))
    return goal_features(ref, torch.as_tensor(HOME_VEC))


def test_plan_goal_matches_library_goal_rover():
    spec = get_robot_spec("rover")
    rp, rq, q = synthetic_walk(4.0)
    clip = body_states_from_pose(spec.load_model(), spec, rp, rq, q, 1 / FPS)
    lib = _clip_goal(clip)[1:-2]  # interior frames (one-sided differences at the ends)
    assert lib.shape[-1] == spec.goal_feat_dim == 2 * NJ + 10
    feat = torch.as_tensor(features_from_pose(rp, rq, q, HOME_VEC))[None]
    plan = plan_goal_frames(feat, torch.as_tensor(HOME_VEC), torch.as_tensor(HOME_VEC))[0, 1:-2]
    j, jv, grav, ang, lin, h = (slice(0, NJ), slice(NJ, 2 * NJ), slice(2 * NJ, 2 * NJ + 3), slice(2 * NJ + 3, 2 * NJ + 6), slice(2 * NJ + 6, 2 * NJ + 9), slice(2 * NJ + 9, 2 * NJ + 10))
    for s, tol in ((j, 1e-9), (jv, 1e-7), (grav, 1e-6), (h, 1e-9), (lin, 5e-3), (ang, 2e-2)):
        err = float((plan[:, s] - lib[:, s]).abs().max())
        assert err < tol, (s, err)
    # Control: the plan one frame late must fail the joint-velocity and joint terms.
    late = plan_goal_frames(torch.roll(feat, 1, dims=1), torch.as_tensor(HOME_VEC), torch.as_tensor(HOME_VEC))[0, 1:-2]
    assert float((late[:, j] - lib[:, j]).abs().max()) > 1e-2
    assert float((late[:, jv] - lib[:, jv]).abs().max()) > 1e-1


@pytest.mark.parametrize("min_norm", [0.15, 0.0])
def test_command_holds_clear_of_the_standing_switch(min_norm):
    cfg = CommandCfg(min_norm=min_norm)
    cmds = RoverCommands(512, cfg, seed=3)
    held = []
    for k in range(50 * 120):
        due = k / FPS >= cmds.t_next  # environments that draw their next hold in this step
        c = cmds.step(k / FPS).copy()
        assert (c >= np.array(cfg.lo) - 1e-12).all() and (c <= np.array(cfg.hi) + 1e-12).all()
        held.append(cmds.dst[due].copy())
    norm = np.linalg.norm(np.concatenate(held), axis=1)
    assert len(norm) > 10_000
    small = ((norm > 0) & (norm < PHASE_THRESHOLD + 0.05)).mean()
    stand = (norm == 0).mean()
    assert 0.08 < stand < 0.25  # p_stand 0.15 of the draws
    if min_norm:
        assert small == 0.0 and norm[norm > 0].min() >= min_norm
    else:
        assert small > 0.005  # control: without the rule, holds inside the switch do occur


def test_foot_centres_are_the_foot_spheres():
    model = get_robot_spec("rover").load_model()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    offs = foot_offsets(model)
    gid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOMS]
    qpos = np.zeros((16, model.nq))
    qpos[:, :3] = rng.normal(size=(16, 3)) * 0.2
    quat = rng.normal(size=(16, 4))
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    qpos[:, 7:] = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1])
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    pos, quat_b, ref = np.zeros((16, model.nbody - 1, 3)), np.zeros((16, model.nbody - 1, 4)), np.zeros((16, 4, 3))
    for k in range(16):
        data.qpos[:] = qpos[k]
        mujoco.mj_kinematics(model, data)
        pos[k], quat_b[k], ref[k] = data.xpos[1:], data.xquat[1:], data.geom_xpos[gid]
    np.testing.assert_allclose(foot_centres(pos, quat_b, names, offs), ref, atol=1e-12)
    # Control: the calf origin (the knee) is a leg length away from the foot.
    assert np.abs(foot_centres(pos, quat_b, names, np.zeros_like(offs)) - ref).max() > 0.2


def test_phase_condition_is_the_policys_clock():
    """The clock the clock-conditioned planner takes: the walking policy's phase input (sin / cos of
    (step * 0.02) mod 0.6 s, zeroed while |command| < 0.1); here its period, its value at a known
    step and the switch."""
    walk = np.tile([[0.5, 0.0, 0.0]], (61, 1))
    steps = np.arange(50, 111)
    ph = phase_condition(walk, steps)
    np.testing.assert_allclose(ph[:31], ph[30:], atol=1e-9)  # 0.6 s = 30 policy steps
    g = (50 * 0.02) % 0.6 / 0.6
    np.testing.assert_allclose(ph[0], [np.sin(2 * np.pi * g), np.cos(2 * np.pi * g)], atol=1e-12)
    assert np.abs(ph[1] - ph[0]).max() > 0.1  # control: a step later the clock has moved
    near = np.array([[0.0999, 0.0, 0.0], [0.1001, 0.0, 0.0], [0.0, 0.06, 0.08], [0.0, 0.0, 0.0]])
    gated = phase_condition(near, np.full(4, 57))
    assert (gated[[0, 3]] == 0).all() and (np.abs(gated[1]).sum() > 0) and (np.abs(gated[2]).sum() > 0)  # |(0, .06, .08)| = 0.1


def test_load_planner_keeps_the_condition_width(tmp_path):
    from supermimic.planner.model import Normalizer, Planner, PlannerCfg
    from supermimic.planner.rover import build_planner, load_planner

    cfg = PlannerCfg(feat_dim=FEAT_DIM, hidden=64, blocks=1)
    norm = Normalizer(torch.zeros(FEAT_DIM), torch.ones(FEAT_DIM), torch.ones(5))
    torch.manual_seed(0)
    m = build_planner(cfg, norm, cond_dim=5)
    torch.nn.init.normal_(m.out[1].weight, std=0.01)  # the output layer starts at zero: make the condition matter
    m.save(tmp_path / "p.pt", cond_dim=5)
    m2, ck = load_planner(tmp_path / "p.pt")
    ctx, cond, g = torch.randn(3, cfg.context, FEAT_DIM), torch.randn(3, 5), torch.zeros(3, dtype=torch.long)
    m.eval()
    torch.testing.assert_close(m2.plan(ctx, cond, g), m.plan(ctx, cond, g))
    assert m2.cond_dim == 5 and ck["cond_dim"] == 5
    assert (m.plan(ctx, cond, g) - m.plan(ctx, cond * torch.tensor([1, 1, 1, -1, -1.0]), g)).abs().max() > 1e-4  # the clock inputs reach the output
    with pytest.raises(RuntimeError):  # control: the base loader builds a 3-wide input and cannot take it
        Planner.load(tmp_path / "p.pt")


def test_command_ramps_respect_the_rate_cap():
    cap = (2.0, 2.0, 3.0)
    for acc_max, ok in ((cap, True), (None, False)):  # control: the uncapped sampler ramps faster
        cfg = CommandCfg(lo=(-1.0, -0.75, -1.0), hi=(1.7, 0.75, 1.0), acc_max=acc_max)
        cmds = RoverCommands(256, cfg, seed=5)
        prev = cmds.step(0.0).copy()
        worst = np.zeros(3)
        for k in range(1, 50 * 60):
            c = cmds.step(k / FPS).copy()
            worst = np.maximum(worst, np.abs(c - prev).max(0) * FPS)
            prev = c
        assert (worst <= np.array(cap) + 1e-9).all() == ok, (acc_max, worst)


def test_actual_velocity_condition_gates_the_clock_on_the_policy_command():
    from supermimic.planner.rover import (
        VelocityMap,
        joystick_cfg,
        joystick_condition,
        monomial_exponents,
        planner_condition,
    )

    e = monomial_exponents(1)
    gain = np.array([0.8, 0.5, 0.9])
    vm = VelocityMap(e, np.diag(gain), np.diag(1 / gain))  # actual = gain * cmd
    np.testing.assert_allclose(vm.command(vm.actual(np.array([[0.3, -0.2, 0.5]]))), [[0.3, -0.2, 0.5]], atol=1e-12)
    ck1 = {"cond_dim": 5}
    ck2 = {"cond_dim": 5, "label": "l1", "velocity_map": vm.to_dict(), "joy_lo": [-0.8, -0.5, -0.9], "joy_hi": [1.5, 0.5, 0.9], "joy_acc": [1.7, 1.4, 2.6]}
    v = np.array([[0.0, 0.07, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.3]])  # 0.07 m/s sideways actual = 0.14 commanded: the policy walks
    steps = np.array([57, 57, 80])
    # command-conditioned checkpoints: the command and the clock, unchanged
    np.testing.assert_array_equal(joystick_condition(ck1, v, steps), planner_condition(v, steps, 5))
    c2 = joystick_condition(ck2, v, steps)
    np.testing.assert_array_equal(c2[:, :3], v)  # the velocity inputs are the joystick (actual units)
    assert np.abs(c2[0, 3:]).sum() > 0 and (c2[1, 3:] == 0).all()  # walking clock at 0.07 actual sideways, none at rest
    assert (planner_condition(v, steps, 5)[0, 3:] == 0).all()  # control: gating on |v| itself would stand
    assert joystick_cfg(ck1).acc_max is None and joystick_cfg(ck2).hi == (1.5, 0.5, 0.9)


def test_stick_ramps_like_the_recording():
    """The deployment stick: every change is a linear ramp of at least 0.2 s (the recording's shortest)
    and within the rate cap, all axes together; a change mid-ramp restarts from where the stick is."""
    from supermimic.planner.rover import RateLimiter

    acc = (1.7, 1.4, 2.6)
    lim = RateLimiter(acc, dt=0.02, n=2)
    out = np.stack([lim(np.array([[1.5, -0.5, 0.9], [0.0, 0.0, -0.25]])) for _ in range(60)])
    steps = np.abs(np.diff(np.concatenate([np.zeros((1, 2, 3)), out]), axis=0))
    assert (steps <= np.array(acc) * 0.02 + 1e-12).all()
    np.testing.assert_allclose(steps[:, 0].max(0)[0], 1.7 * 0.02)  # the long axis sets the ramp (1.5 / 1.7 = 0.88 s) ...
    assert np.argmax(np.abs(out[:, 0, 0] - 1.5) < 1e-9) == 44  # ... and every axis gets there together (0.88 s: the 45th step)
    assert np.argmax(np.abs(out[:, 0, 1] + 0.5) < 1e-9) == 44
    # a small change (a slow turn released) still takes 0.2 s = 10 steps, not 0.25 / 2.6 = 0.1 s
    lim2 = RateLimiter(acc, dt=0.02, n=1)
    for _ in range(50):
        lim2(np.array([[0.0, 0.0, -0.25]]))
    rel = np.array([lim2(np.zeros((1, 3)))[0, 2] for _ in range(12)])
    assert np.abs(rel[8]) > 1e-3 and np.abs(rel[9]) < 1e-12
    np.testing.assert_allclose(np.diff(np.r_[-0.25, rel[:10]]), 0.025)  # linear
    # control: a bare rate limit at the same cap is done in 5 steps
    bare = -0.25 + np.minimum(np.arange(1, 13) * 2.6 * 0.02, 0.25)
    assert np.argmax(np.abs(bare) < 1e-12) == 4
    # retarget mid-ramp: continuous, then a new ramp from there
    lim3 = RateLimiter(acc, dt=0.02, n=1)
    a = [lim3(np.array([[0.0, 0.0, 0.5]]))[0, 2] for _ in range(5)]
    b = [lim3(np.zeros((1, 3)))[0, 2] for _ in range(12)]
    assert abs(b[0] - a[-1]) <= 2.6 * 0.02 + 1e-12 and abs(b[9]) < 1e-12 and abs(b[8]) > 1e-3
