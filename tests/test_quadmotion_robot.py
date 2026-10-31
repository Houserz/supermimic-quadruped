"""The retargeting's own kinematics against MuJoCo's, and the mirror map against a reflection,
on every robot config in ``configs/robots`` (the Rover X1 and, as a second robot, mjlab's Go1).

Both failures would be silent downstream: an IK that disagrees with the model by a constant
offset still produces smooth clips, just with every foot in the wrong place; a mirror map with
one sign wrong produces a plausible-looking clip whose feet cross.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quadmotion import dog, robot
from quadmotion.clip import body_states_from_pose, mirror_pose

ROBOTS = tuple(sorted(p.stem for p in robot.ROBOT_CONFIGS.glob("*.yaml")))  # every configs/robots/*.yaml
RNG = np.random.default_rng(0)
AI4A = Path(__file__).resolve().parents[1] / "data" / "raw" / "ai4animation_dog" / "bvh"


@pytest.fixture(params=ROBOTS)
def quad(request) -> robot.Quadruped:
    return robot.load_robot(request.param)


def _random_q(quad: robot.Quadruped, n: int) -> np.ndarray:
    """Joint angles (model order) inside the limits on the knee-backwards branch, away from the straight leg."""
    q = np.zeros((n, 12))
    for l in robot.LEGS:
        a, t, c = quad.leg_q[l]
        q[:, a] = RNG.uniform(-0.6, 0.6, n)
        q[:, t] = RNG.uniform(max(quad.q_lo[t], -1.0), min(quad.q_hi[t], 2.0), n)
        q[:, c] = RNG.uniform(max(quad.q_lo[c], -2.4), min(quad.q_hi[c], -0.15), n)
    return q


def _mujoco_feet(quad: robot.Quadruped, root_pos, root_quat, q) -> np.ndarray:
    """(T, 4, 3) foot sites in LEGS order; q in model order."""
    model = quad.model
    data = mujoco.MjData(model)
    out = np.zeros((len(q), 4, 3))
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in quad.foot_sites]
    for k in range(len(q)):
        data.qpos[:3] = root_pos[k]
        data.qpos[3:7] = root_quat[k]
        data.qpos[7:] = q[k]
        mujoco.mj_kinematics(model, data)
        out[k] = data.site_xpos[sites]
    return out


def test_analytic_fk_matches_mujoco(quad):
    q = _random_q(quad, 200)
    root = np.zeros((200, 3))
    quat = np.tile([1.0, 0, 0, 0], (200, 1))
    ref = _mujoco_feet(quad, root, quat, q)
    assert np.abs(robot.feet_in_trunk(quad, q) - ref).max() < 1e-12


def test_ik_reaches_every_reachable_foot(quad):
    """Any foot the model can reach is reached (the joint solution may be the other abad branch)."""
    q = _random_q(quad, 500)
    feet = robot.feet_in_trunk(quad, q)
    _, resid, clipped = robot.solve_ik(quad, {l: feet[:, i] for i, l in enumerate(robot.LEGS)})
    assert resid.max() < 1e-9 and clipped.max() == 0.0


def test_ik_inverts_fk_for_feet_below_the_thigh_joint(quad):
    q = _random_q(quad, 2000)
    feet = robot.feet_in_trunk(quad, q)
    # Keep the samples whose foot is below the thigh joint in the leg plane (the stance branch).
    below = np.ones(len(q), bool)
    for l in robot.LEGS:
        g = quad.legs[l]
        qq = q[:, quad.leg_q[l]]
        zz = -g.l_thigh * np.cos(qq[:, 1]) - g.l_calf * np.cos(qq[:, 1] + qq[:, 2])
        below &= zz < -0.02
    q, feet = q[below], feet[below]
    assert len(q) > 300
    for i, l in enumerate(robot.LEGS):
        q_ik = robot.leg_ik(quad.legs[l], feet[:, i], abad_limit=float(quad.q_hi[quad.leg_q[l][0]]))
        assert np.abs(q_ik - q[:, quad.leg_q[l]]).max() < 1e-7


def test_sequence_ik_does_not_jump_when_a_paw_rises_past_the_hip():
    """A front paw swept from stance to above the hip: the per-frame rule switches branch where
    the lower branch leaves the abad range and the abad angle jumps; the sequence solver must not."""
    quad = robot.load_robot("rover")
    g = quad.legs["FL"]
    t = np.linspace(0.0, 1.0, 400)
    q_path = np.stack([0.3 * np.ones_like(t), 0.84 + 1.6 * t, -1.3 + 0.9 * t], -1)  # thigh up to 2.44 rad
    p = robot.leg_fk(g, q_path)
    feet = {l: robot.leg_fk(quad.legs[l], np.tile([0.0, 0.84, -1.3], (len(t), 1))) for l in robot.LEGS}
    feet["FL"] = p
    q, resid, _ = robot.solve_ik(quad, feet)
    assert resid[:, 0].max() < 1e-9
    assert np.abs(np.diff(q[:, quad.leg_q["FL"]], axis=0)).max() < 0.05
    # Control: the per-frame rule does jump on this path.
    q_rule = robot.leg_ik(g, p, abad_limit=float(quad.q_hi[0]))
    assert np.abs(np.diff(q_rule, axis=0)).max() > 0.3


def test_ik_raised_paw_takes_the_upper_branch():
    """A front paw swung forward above the hip (high five) is only reachable with zz > 0."""
    quad = robot.load_robot("rover")
    g = quad.legs["FL"]
    q = np.array([0.1, 2.3, -0.6])
    p = robot.leg_fk(g, q)
    q_ik = robot.leg_ik(g, p, abad_limit=0.6632)
    assert abs(q_ik[0]) < 0.6632 and np.allclose(robot.leg_fk(g, q_ik), p, atol=1e-9)


def test_ik_branch_is_knees_backwards(quad):
    """Control group for the tests above: the other branch reaches the same feet with calf > 0."""
    g = quad.legs["FL"]
    lim = float(quad.q_hi[quad.leg_q["FL"][0]])
    p = robot.leg_fk(g, np.array([0.0, *quad.stance]))
    q_back = robot.leg_ik(g, p, abad_limit=lim)
    q_fwd = robot.leg_ik(g, p, abad_limit=lim, knee_sign=+1.0)
    assert q_back[2] < 0 < q_fwd[2]
    assert np.allclose(robot.leg_fk(g, q_fwd), p) and robot.knee_pos(g, q_back)[0] < g.abad_pos[0]


def test_mirror_reflects_every_foot(quad):
    n = 50
    q = _random_q(quad, n)
    root = RNG.normal(0, 0.3, (n, 3))
    quat = Rotation.from_rotvec(RNG.normal(0, 0.4, (n, 3))).as_quat(scalar_first=True)
    feet = _mujoco_feet(quad, root, quat, q)
    mr, mq, mjp = mirror_pose(quad, root, quat, q)
    feet_m = _mujoco_feet(quad, mr, mq, mjp)
    swap = [robot.LEGS.index(robot.MIRROR_LEG[l]) for l in robot.LEGS]
    assert np.abs(feet_m[:, swap] * [1, -1, 1] - feet).max() < 1e-9
    # Control: without the abad sign flip the reflection fails.
    bad = q[:, list(quad.mirror_perm)]
    feet_bad = _mujoco_feet(quad, mr, mq, bad)
    assert np.abs(feet_bad[:, swap] * [1, -1, 1] - feet).max() > 1e-2


def test_body_states_layout(quad):
    q = np.zeros((5, 12))
    for l in robot.LEGS:
        q[:, quad.leg_q[l][1:]] = quad.stance
    root = np.tile([0.0, 0.0, quad.nominal_root_height()], (5, 1))
    quat = np.tile([1.0, 0, 0, 0], (5, 1))
    c = body_states_from_pose(quad.model, root, quat, q)
    assert c.body_names[0] == quad.root_body and len(c.body_names) == 13
    assert c.joint_names == list(quad.joint_names)
    # Standing pose: feet at the foot radius above the ground.
    feet = _mujoco_feet(quad, root, quat, q)
    assert np.allclose(feet[:, :, 2], quad.foot_radius, atol=1e-9)


@pytest.mark.skipif(not AI4A.exists(), reason="run scripts/data/fetch_sources.py first")
def test_dog_clip_retargets_onto_every_robot(quad):
    """End to end on one AI4Animation walk: most frames pass every check, the feet stay out of
    the floor, and the clip's FK agrees with MuJoCo's."""
    skel = dog.AI4ANIMATION
    k = dog.load_keypoints(AI4A / "D1_001_KAN01_001.bvh", skel)
    cal = dog.calibrate([k], skel, quad.nominal_root_height())
    r = dog.retarget(k, skel, cal, quad, checks=dog.checks_for(quad))
    assert r.valid.mean() > 0.8, {n: int(m.sum()) for n, m in r.fail.items() if m.any()}
    feet = _mujoco_feet(quad, r.root_pos, r.root_quat, r.joint_pos)
    assert feet[r.valid, :, 2].min() > quad.foot_radius - 2e-3
