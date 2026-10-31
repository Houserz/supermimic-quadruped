"""Rover skills (planner.rover_skills): leg IK, placement, the skill sequences, the driver's state machine."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.planner import rover_skills as rs
from supermimic.planner.rover import FEAT_DIM, HOME_VEC

RELEASED = rs.ROOT / "logs" / "planner" / "rover_v2_l1" / "planner_last.pt"  # carries the hub frame


def _hub():
    if RELEASED.exists():
        return np.asarray(torch.load(RELEASED, map_location="cpu", weights_only=False)["hub"], np.float64)
    return rs.hub_frame()


need_data = pytest.mark.skipif(not (all((rs.LIBRARY_DIR / f"{c}.npz").exists() for sd in rs.SKILLS.values() for c in sd.clips)
                                    and (RELEASED.exists() or rs.PLANNER_DATA.is_dir())),
                               reason="needs the built library (data/npz/rover) and the released planner (or its walking data)")
SL = rs.SL


@pytest.fixture(scope="module")
def kin():
    return rs.LegKinematics()


@pytest.fixture(scope="module")
def lib(kin):
    return rs.SkillLibrary(kin=kin, hub=_hub())


def test_ik_reaches_fk_targets(kin):
    """IK from HOME reaches the feet of random poses near standing (control: HOME's own feet are off)."""
    rng = np.random.default_rng(0)
    pos, quat = np.array([0.0, 0.0, 0.38]), np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        q = HOME_VEC + rng.uniform(-0.3, 0.3, 12)
        target = kin.feet(pos, quat, q)
        assert np.abs(kin.feet(pos, quat, HOME_VEC) - target).max() > 1e-3
        q_ik, err = kin.ik(pos, quat, target, HOME_VEC.copy())
        assert err < 1e-5
        assert np.abs(kin.feet(pos, quat, q_ik) - target).max() < 1e-5


def test_fit_rigid_2d():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(4, 3))
    yaw, xy = 0.4, np.array([0.3, -0.2])
    c, s = np.cos(yaw), np.sin(yaw)
    dst = src.copy()
    dst[:, 0], dst[:, 1] = c * src[:, 0] - s * src[:, 1] + xy[0], s * src[:, 0] + c * src[:, 1] + xy[1]
    y2, xy2 = rs.fit_rigid_2d(src, dst)
    assert abs(y2 - yaw) < 1e-9 and np.allclose(xy2, xy)


@need_data
def test_sequences_start_at_start_and_end_on_hub(lib, kin):
    hub = lib.hub
    for name in lib.names:
        feat, seg, _ = lib.sequence(name, hub)
        assert np.isfinite(feat).all() and feat.shape[1] == FEAT_DIM
        assert seg[0] == rs.BLEND_IN and seg[-1] == rs.HUB and rs.CLIP in seg
        # continuity with the start frame, and the planner context at the end is the hub, standing still
        assert np.abs(feat[0, SL["joint"]] - hub[SL["joint"]]).max() < 0.05
        tail = feat[-int(lib.cfg.hub_hold_s * rs.FPS) :]
        for k in ("joint", "height", "grav"):
            assert np.abs(tail[:, SL[k]] - hub[SL[k]]).max() < 1e-5, (name, k)
        assert np.abs(tail[:, SL["lin_h"]]).max() < 1e-6 and np.abs(tail[:, SL["yaw_rate"]]).max() < 1e-6
        # blends keep the feet on the floor (no foot centre deeper than 5 mm below its stance height)
        from supermimic.planner.features import pose_from_features

        rp, rq, jq = pose_from_features(np.concatenate([hub[None], feat]), HOME_VEC)
        z = np.array([kin.feet(rp[i], rq[i], jq[i])[:, 2] for i in range(len(rp))])
        blend = np.isin(np.concatenate([[rs.STOP], seg]), (rs.BLEND_IN, rs.BLEND_OUT, rs.INNER))
        assert z[blend].min() > 0.028 - 0.005, name


@need_data
def test_ik_blend_plants_feet_joint_blend_does_not(kin):
    """The sit's blend out (sitting -> standing): IK keeps the feet within 2 cm of their path's ends;
    joint-space interpolation drops a foot well into the floor on the way (the control)."""
    from supermimic.planner.features import pose_from_features

    depth = {}
    for method in ("ik", "joint"):
        lib = rs.SkillLibrary({"sit": rs.SKILLS["sit"]}, cfg=rs.SkillCfg(method=method), kin=kin, hub=_hub())
        feat, seg, _ = lib.sequence("sit", lib.hub)
        rp, rq, jq = pose_from_features(feat, HOME_VEC)
        z = np.array([kin.feet(rp[i], rq[i], jq[i])[:, 2] for i in np.flatnonzero(seg == rs.BLEND_OUT)])
        depth[method] = z.min()
    assert depth["ik"] > 0.025 and depth["joint"] < 0.015, depth


class _StandingPlanner:
    """Plans the last context frame forever (a planner that stands), with the real planner's interface."""

    cfg = SimpleNamespace(context=25, horizon=50)
    cond_dim = 5

    def plan(self, ctx, cond, gait):
        return ctx[:, -1:].repeat(1, self.cfg.horizon, 1)


@need_data
def test_driver_stops_plays_and_resumes(lib):
    n = 3
    drv = rs.SkillDriver(_StandingPlanner(), {"replan": 5}, lib, n, torch.tensor(HOME_VEC, dtype=torch.float32), "cpu")
    drv.reset(np.repeat(lib.hub[None, None], n, 0).repeat(25, 1), np.zeros(n, int))
    zero = np.zeros((n, 3))
    for _ in range(7):
        g = drv.step(zero)
    assert g.shape == (n, 10 * (2 * 12 + 10))
    assert drv.request(1, "stretch") and not drv.request(1, "sit"), "a second request while busy is ignored"
    feat, _, _ = lib.sequence("stretch", lib.hub)
    modes, segs = [], []
    for _ in range(len(feat) + 40):
        drv.step(zero)
        modes.append(drv.mode_now[1])
        segs.append(drv.seg_now[1])
    modes, segs = np.array(modes), np.array(segs)
    ev = drv.events[0]
    assert ev["skill"] == "stretch" and not ev["timeout"]
    # stopping lasts until ready_replans replans have seen a standing plan, then exactly the sequence plays
    assert 0 < ev["stop_frames"] <= (lib.cfg.ready_replans + 1) * 5
    assert (modes == rs.M_SKILL).sum() == len(feat)
    assert rs.CLIP in segs and segs[-1] in (rs.WALK, rs.STAND) and modes[-1] == rs.M_WALK  # STAND: the stand guard holds the hub
    assert (drv.mode[[0, 2]] == rs.M_WALK).all() and len(drv.events) == 1
    # back to walking from the hub: the planner's context is the hub frame
    assert torch.allclose(drv.hist[1, -1], torch.tensor(lib.hub, dtype=torch.float32), atol=1e-5)


class _WakingPlanner(_StandingPlanner):
    """Stands for its first ``quiet`` calls, then plans a yaw oscillation in place (the planner's
    standing state is not a fixed point: the stand guard's rescue)."""

    def __init__(self, quiet: int):
        self.calls, self.quiet = 0, quiet

    def plan(self, ctx, cond, gait):
        self.calls += 1
        out = ctx[:, -1:].repeat(1, self.cfg.horizon, 1).clone()
        if self.calls > self.quiet:
            out[..., rs.SL["yaw_rate"]] = 0.5 * (-1) ** self.calls
        return out


@need_data
@pytest.mark.parametrize("hold", [True, False])
def test_stand_guard_holds_and_releases(lib, hold):
    """With the guard, a zero stick on a standing reference stops replanning: the reference stays at
    the held frame although the planner would start to rock; a moving stick hands back to the planner.
    Without the guard (the control) the rocking reaches the reference."""
    cfg = rs.SkillCfg(stand_hold=hold)
    lib2 = rs.SkillLibrary({}, cfg=cfg, hub=lib.hub, kin=lib.kin)
    planner = _WakingPlanner(quiet=12)
    drv = rs.SkillDriver(planner, {"replan": 5}, lib2, 1, torch.tensor(HOME_VEC, dtype=torch.float32), "cpu")
    drv.reset(np.repeat(lib.hub[None, None], 25, 1), np.zeros(1, int))
    wz = []
    for _ in range(150):
        drv.step(np.zeros((1, 3)))
        wz.append(float(drv.hist[0, -1, SL["yaw_rate"]][0]))
    if hold:
        assert drv.held[0] and drv.seg_now[0] == rs.STAND and planner.calls <= 12
        assert max(abs(w) for w in wz) == 0.0
        drv.step(np.array([[0.3, 0.0, 0.0]]))
        assert not drv.held[0]
        calls = planner.calls
        for _ in range(5):
            drv.step(np.array([[0.3, 0.0, 0.0]]))
        assert planner.calls == calls + 1 and drv.seg_now[0] == rs.WALK
    else:
        assert not drv.held.any() and max(abs(w) for w in wz) > 0.4


@need_data
def test_stand_rescue_blends_a_rocking_plan_to_the_hub(lib):
    """A planner that rocks from the first call never gives a quiet reference to hold; after
    ``stand_rescue_s`` of zero stick the guard blends to the hub and holds it (control: rescue off keeps rocking)."""
    for rescue in (1.0, None):
        cfg = rs.SkillCfg(stand_hold=True, stand_rescue_s=rescue)
        lib2 = rs.SkillLibrary({}, cfg=cfg, hub=lib.hub, kin=lib.kin)
        drv = rs.SkillDriver(_WakingPlanner(quiet=0), {"replan": 5}, lib2, 1, torch.tensor(HOME_VEC, dtype=torch.float32), "cpu")
        ctx = np.repeat(lib.hub[None, None], 25, 1).copy()
        ctx[0, :, SL["yaw_rate"]] = 0.5 * (-1.0) ** np.arange(25)[:, None]  # already rocking: nothing quiet to hold
        drv.reset(ctx, np.zeros(1, int))
        segs, wz = [], []
        for _ in range(250):
            drv.step(np.zeros((1, 3)))
            segs.append(int(drv.seg_now[0]))
            wz.append(float(drv.hist[0, -1, SL["yaw_rate"]][0]))
        if rescue:
            assert len(drv.rescues) == 1 and rs.RESCUE in segs and segs[-1] == rs.STAND and drv.held[0]
            assert max(abs(w) for w in wz[-50:]) < 1e-6
        else:
            assert not drv.rescues and max(abs(w) for w in wz[-50:]) > 0.2  # the crossfade halves the planner's ±0.5
