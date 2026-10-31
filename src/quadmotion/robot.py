"""A 12-DoF quadruped as the retargeting sees it: model, joint order, leg geometry, analytic IK.

The robot is described by a small YAML file in ``configs/robots/`` (see ``rover.yaml`` and
``go1.yaml``); everything geometric is read from its MJCF rather than typed in, so a model update
changes the numbers here too. Supported legs (most 12-DoF quadrupeds: Rover X1, Go1, Go2, A1,
...), checked when the model is loaded:

    abad    hinge about +x at the hip body origin
    thigh   hinge about +y, its body offset from the hip body only sideways (0, +-d, 0)
    calf    hinge about +y, its body straight below the thigh joint (0, 0, -l_thigh)
    foot    a site straight below the calf joint (0, 0, -l_calf); a sphere geom gives its radius

**Knee direction.** ``knee_sign`` picks the IK branch: -1 = calf < 0, knees pointing backwards on
every leg (Rover X1, Go1 and most Unitree dogs).

Internally the legs are always handled in the order ``LEGS`` = (FL, FR, RL, RR); joint arrays
(``q``, clips) follow the *model's* joint order, whatever it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ROBOT_CONFIGS = ROOT / "configs" / "robots"

LEGS = ("FL", "FR", "RL", "RR")
# Left-right mirror (reflection through the trunk's x-z plane): legs swap sides, the abad hinge
# (about x) changes sign, the pitch hinges (about y) keep it. Checked by FK in the tests.
MIRROR_LEG = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}


@dataclass(frozen=True)
class LegGeometry:
    abad_pos: np.ndarray  # (3,) abad joint in the trunk frame
    thigh_offset: float  # signed lateral offset of the thigh joint from the abad joint
    l_thigh: float
    l_calf: float  # calf joint -> foot centre

    @property
    def thigh_pos(self) -> np.ndarray:
        return self.abad_pos + np.array([0.0, self.thigh_offset, 0.0])


@dataclass(frozen=True)
class Quadruped:
    name: str
    xml: Path
    model: mujoco.MjModel
    root_body: str
    joint_names: tuple[str, ...]  # model order (qpos order after the free joint)
    leg_q: dict[str, np.ndarray]  # leg -> indices of (abad, thigh, calf) in joint_names
    foot_sites: tuple[str, ...]  # LEGS order
    legs: dict[str, LegGeometry]
    foot_radius: float
    q_lo: np.ndarray  # (12,) model order
    q_hi: np.ndarray
    trunk_half: np.ndarray  # trunk collision box half sizes
    trunk_box_pos: np.ndarray
    knee_sign: float
    stance: tuple[float, float]  # (thigh, calf) of the nominal stance, front legs
    mirror_perm: tuple[int, ...]  # model order
    mirror_sign: np.ndarray
    checks: dict = field(default_factory=dict)  # overrides of ``dog.Checks`` for this robot

    @property
    def hip_x(self) -> float:
        return float(self.legs["FL"].abad_pos[0])

    def nominal_root_height(self, thigh: float | None = None, calf: float | None = None) -> float:
        """Root height standing on flat ground with every leg at (0, thigh, calf)."""
        thigh = self.stance[0] if thigh is None else thigh
        calf = self.stance[1] if calf is None else calf
        g = self.legs["FL"]
        z = g.l_thigh * np.cos(thigh) + g.l_calf * np.cos(thigh + calf)
        return float(z + self.foot_radius)


def resolve_xml(path: str) -> Path:
    """``mjlab:<path>`` is relative to the installed mjlab package; anything else to the repo root."""
    if path.startswith("mjlab:"):
        import mjlab

        return Path(mjlab.__file__).parent / path.removeprefix("mjlab:")
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def body_names(model: mujoco.MjModel) -> list[str]:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    assert names[0] == "world"
    return names[1:]


def model_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Hinge joints in qpos order (the free joint, joint 0, left out)."""
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "the first joint must be the floating base"
    return tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt))


def load_robot(name_or_path: str | Path = "rover") -> Quadruped:
    """Load ``configs/robots/<name>.yaml`` (or a YAML path) and its MJCF."""
    p = Path(name_or_path)
    cfg_path = p if p.suffix in (".yaml", ".yml") else ROBOT_CONFIGS / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    xml = resolve_xml(cfg["xml"])
    model = mujoco.MjModel.from_xml_path(str(xml))
    joint_names = model_joint_names(model)
    assert len(joint_names) == 12 and all(model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE), "expects 12 hinge joints"

    def jid(n: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        assert i >= 0, f"{cfg_path.name}: no joint {n!r} in {xml.name}"
        return i

    def gid(n: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        assert i >= 0, f"{cfg_path.name}: no geom {n!r} in {xml.name}"
        return i

    legs, leg_q, sites = {}, {}, []
    for l in LEGS:
        abad, thigh, calf, site_name = cfg["legs"][l]
        ja, jt, jc = jid(abad), jid(thigh), jid(calf)
        for j, axis in ((ja, (1, 0, 0)), (jt, (0, 1, 0)), (jc, (0, 1, 0))):
            assert np.allclose(model.jnt_axis[j], axis), f"{l}: joint {model.joint(j).name} axis {model.jnt_axis[j]}, expected {axis}"
            assert np.allclose(model.jnt_pos[j], 0), f"{l}: joint {model.joint(j).name} not at its body origin"
        hip_b, thigh_b, calf_b = (int(model.jnt_bodyid[j]) for j in (ja, jt, jc))
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        assert site >= 0, f"{cfg_path.name}: no site {site_name!r}"
        assert model.site_bodyid[site] == calf_b, f"{l}: foot site must be on the calf body"
        assert np.allclose(model.body_pos[thigh_b][[0, 2]], 0), f"{l}: thigh body offset must be sideways only"
        assert np.allclose(model.body_pos[calf_b][:2], 0), f"{l}: calf body must sit straight below the thigh joint"
        assert np.allclose(model.site_pos[site][:2], 0), f"{l}: foot site must sit straight below the calf joint"
        legs[l] = LegGeometry(
            abad_pos=model.body_pos[hip_b].copy(),
            thigh_offset=float(model.body_pos[thigh_b][1]),
            l_thigh=float(-model.body_pos[calf_b][2]),
            l_calf=float(-model.site_pos[site][2]),
        )
        leg_q[l] = np.array([joint_names.index(n) for n in (abad, thigh, calf)])
        sites.append(site_name)

    # Mirror map in model order: each joint takes its partner leg's joint of the same kind; abad negates.
    perm = [0] * 12
    sign = np.ones(12)
    for l in LEGS:
        for k in range(3):
            perm[int(leg_q[l][k])] = int(leg_q[MIRROR_LEG[l]][k])
            sign[int(leg_q[l][k])] = -1.0 if k == 0 else 1.0

    geoms = cfg["geoms"]
    foot, trunk = gid(geoms["foot"]), gid(geoms["trunk"])
    assert model.geom_type[trunk] == mujoco.mjtGeom.mjGEOM_BOX, "trunk geom must be a box"
    jids = [jid(n) for n in joint_names]
    return Quadruped(
        name=cfg_path.stem,
        xml=xml,
        model=model,
        root_body=cfg["root_body"],
        joint_names=joint_names,
        leg_q=leg_q,
        foot_sites=tuple(sites),
        legs=legs,
        foot_radius=float(model.geom_size[foot][0]),
        q_lo=model.jnt_range[jids, 0].copy(),
        q_hi=model.jnt_range[jids, 1].copy(),
        trunk_half=model.geom_size[trunk].copy(),
        trunk_box_pos=model.geom_pos[trunk].copy(),
        knee_sign=float(cfg.get("knee_sign", -1.0)),
        stance=tuple(float(x) for x in cfg["stance"]),
        mirror_perm=tuple(perm),
        mirror_sign=sign,
        checks=dict(cfg.get("checks") or {}),
    )


# ------------------------------------------------------------------------------------ IK / FK


def leg_fk(g: LegGeometry, q: np.ndarray) -> np.ndarray:
    """Foot centre in the trunk frame for joint angles q (..., 3) = (abad, thigh, calf)."""
    q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2]
    x = -g.l_thigh * np.sin(q2) - g.l_calf * np.sin(q2 + q3)
    z = -g.l_thigh * np.cos(q2) - g.l_calf * np.cos(q2 + q3)
    y = np.full_like(x, g.thigh_offset)
    c, s = np.cos(q1), np.sin(q1)
    return np.stack([x, y * c - z * s, y * s + z * c], -1) + g.abad_pos


def knee_pos(g: LegGeometry, q: np.ndarray) -> np.ndarray:
    """Calf joint (knee) in the trunk frame."""
    q1, q2 = q[..., 0], q[..., 1]
    x = -g.l_thigh * np.sin(q2)
    z = -g.l_thigh * np.cos(q2)
    y = np.full_like(x, g.thigh_offset)
    c, s = np.cos(q1), np.sin(q1)
    return np.stack([x, y * c - z * s, y * s + z * c], -1) + g.abad_pos


def leg_ik_branches(g: LegGeometry, p: np.ndarray, knee_sign: float = -1.0) -> tuple[np.ndarray, np.ndarray]:
    """Both abad branches for foot targets p (..., 3): (foot below the thigh joint, foot above)."""
    r = p - g.abad_pos
    px, py, pz = r[..., 0], r[..., 1], r[..., 2]
    d = g.thigh_offset
    zz_abs = np.sqrt(np.maximum(py**2 + pz**2 - d * d, 1e-12))
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    out = []
    for zz in (-zz_abs, zz_abs):
        q1 = wrap(np.arctan2(pz, py) - np.arctan2(zz, d))
        l1, l2 = g.l_thigh, g.l_calf
        c3 = np.clip((px**2 + zz**2 - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0)
        q3 = knee_sign * np.arccos(c3)
        q2 = wrap(np.arctan2(-px, -zz) - np.arctan2(l2 * np.sin(q3), l1 + l2 * np.cos(q3)))
        out.append(np.stack([q1, q2, q3], -1))
    return out[0], out[1]


def leg_ik(g: LegGeometry, p: np.ndarray, abad_limit: float, knee_sign: float = -1.0) -> np.ndarray:
    """Closed-form IK: foot centre p (..., 3) in the trunk frame -> (abad, thigh, calf).

    Unreachable targets are projected onto the workspace (leg straightened, or the foot pulled
    out of the cylinder of radius |thigh_offset| about the abad axis); the caller measures the
    residual with ``leg_fk`` and decides whether the frame is usable.
    """
    r = p - g.abad_pos
    px, py, pz = r[..., 0], r[..., 1], r[..., 2]
    d = g.thigh_offset
    rho2 = py**2 + pz**2
    # In the abad-rotated frame the foot is at (d, zz); zz < 0 is the foot below the thigh joint,
    # zz > 0 a leg swung up past horizontal (a raised paw). Both can reach the same point. We take
    # the foot-below solution unless its abad angle is outside the model's range, which is exactly
    # the raised-paw case (its abad is then near pi).
    zz_abs = np.sqrt(np.maximum(rho2 - d * d, 1e-12))
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    q1_dn = wrap(np.arctan2(pz, py) - np.arctan2(-zz_abs, d))
    q1_up = wrap(np.arctan2(pz, py) - np.arctan2(zz_abs, d))
    up = (np.abs(q1_dn) > abad_limit) & (np.abs(q1_up) < np.abs(q1_dn))
    q1 = np.where(up, q1_up, q1_dn)
    zz = np.where(up, zz_abs, -zz_abs)
    L2 = px**2 + zz**2
    l1, l2 = g.l_thigh, g.l_calf
    c3 = np.clip((L2 - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0)
    q3 = knee_sign * np.arccos(c3)
    A = l1 + l2 * np.cos(q3)
    B = l2 * np.sin(q3)
    q2 = np.arctan2(-px, -zz) - np.arctan2(B, A)
    q2 = (q2 + np.pi) % (2 * np.pi) - np.pi
    return np.stack([q1, q2, q3], -1)


def solve_ik(robot: Quadruped, feet_b: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IK for all legs. feet_b[leg]: (T, 3) foot centres in the trunk frame.

    Returns joint angles (T, 12) in model order clipped to the model limits, the per-leg residual
    of the clipped solution (T, 4, LEGS order) in metres, and the amount of clipping (T, 12) in radians.
    """
    T = next(iter(feet_b.values())).shape[0]
    q = np.zeros((T, 12))
    for l in LEGS:
        idx = robot.leg_q[l]
        lo, hi = robot.q_lo[idx], robot.q_hi[idx]
        branches = [_into_range(b, lo, hi) for b in leg_ik_branches(robot.legs[l], feet_b[l], robot.knee_sign)]
        q[:, idx] = _continuous_branch(*branches, lo, hi)
    q_clip = np.clip(q, robot.q_lo, robot.q_hi)
    clip_amount = np.abs(q - q_clip)
    resid = np.zeros((T, 4))
    for i, l in enumerate(LEGS):
        resid[:, i] = np.linalg.norm(leg_fk(robot.legs[l], q_clip[:, robot.leg_q[l]]) - feet_b[l], axis=-1)
    return q_clip, resid, clip_amount


def _into_range(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Pitch angles come out of the IK in [-pi, pi); a model whose thigh or calf range reaches past
    pi (Go1's thigh goes to 4.5 rad) gets the 2 pi-shifted angle where that one is in range."""
    out = q.copy()
    for k in (1, 2):
        for shift in (2 * np.pi, -2 * np.pi):
            moved = out[:, k] + shift
            fix = ((out[:, k] < lo[k]) | (out[:, k] > hi[k])) & (moved >= lo[k]) & (moved <= hi[k])
            out[fix, k] = moved[fix]
    return out


def _continuous_branch(q_dn: np.ndarray, q_up: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per frame, the branch inside the joint ranges closest to the previous frame's choice.

    Choosing per frame by a fixed rule switches branches where the rule flips, not where the two
    branches meet, and the abad angle jumps there (a raised paw read as 0.5 rad in one frame).
    A branch counts as inside when all three joints are (a narrow thigh range, as on the Go1, can
    put the other branch out of reach); if neither is, both stay candidates and the caller clips."""
    T = len(q_dn)
    out = np.empty_like(q_dn)
    ok_dn = ((q_dn >= lo) & (q_dn <= hi)).all(1)
    ok_up = ((q_up >= lo) & (q_up <= hi)).all(1)
    prev = q_dn[0] if ok_dn[0] or not ok_up[0] else q_up[0]
    for t in range(T):
        cands = [c for c, ok in ((q_dn[t], ok_dn[t]), (q_up[t], ok_up[t])) if ok] or [q_dn[t], q_up[t]]
        prev = min(cands, key=lambda c: float(np.abs(c - prev).sum()))
        out[t] = prev
    return out


def feet_in_trunk(robot: Quadruped, q: np.ndarray) -> np.ndarray:
    """(T, 4, 3) foot centres in the trunk frame (LEGS order) for joint angles (T, 12, model order)."""
    return np.stack([leg_fk(robot.legs[l], q[:, robot.leg_q[l]]) for l in LEGS], 1)


def knees_in_trunk(robot: Quadruped, q: np.ndarray) -> np.ndarray:
    return np.stack([knee_pos(robot.legs[l], q[:, robot.leg_q[l]]) for l in LEGS], 1)


def trunk_corners(robot: Quadruped) -> np.ndarray:
    """(8, 3) corners of the trunk collision box in the trunk frame."""
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return robot.trunk_box_pos + s * robot.trunk_half
