"""Rover X1 pieces of the walking planner (docs/PLANNER.md).

Everything robot-independent is in the other modules: the frame features and their inverse
(``features.py``: for the Rover 12 + 1 + 3 + 3 + 1 = 20 per frame), the dataset loader
(``dataset.py``), the model (``model.py``) and the plan -> tracker-goal conversion (``goal.py``).
This module adds what is the Rover's own:

* ``HOME_VEC``: the default pose in ``rover.JOINT_NAMES`` order, the features' joint offset;
* the feet: spheres on the calves (no foot bodies), so a foot is the calf body's frame plus the
  foot geom's offset, read from the model (``foot_centres``);
* ``RoverCommands``: the joystick-like command stream the walking data was recorded with, used
  unchanged by the closed loop, so both draw from one distribution;
* the walking policy's gait clock and the velocity map of the actual-velocity planners.

The RL walking policy the data comes from (trained locally) stands still whenever the norm of its
command (vx, vy, wz) is below ``PHASE_THRESHOLD``: its gait-phase input is zeroed there. A held command is therefore either exactly zero or at least ``min_norm`` (default 0.15) in
that norm, so no hold sits on the switch; ramps between holds cross it as a joystick does, and the
planner learns the policy's own response there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from supermimic.planner.features import feat_dim
from supermimic.robots.rover import HOME, JOINT_NAMES, LEGS

NJ = len(JOINT_NAMES)
HOME_VEC = np.array([HOME[j] for j in JOINT_NAMES])
FEAT_DIM = feat_dim(NJ)  # 20
_WORD = {"FL": "front_left", "FR": "front_right", "RL": "rear_left", "RR": "rear_right"}
FOOT_GEOMS: tuple[str, ...] = tuple(f"{leg}_foot_collision" for leg in LEGS)
FOOT_BODIES: tuple[str, ...] = tuple(f"link_{_WORD[leg]}_calf" for leg in LEGS)
PHASE_THRESHOLD = 0.1  # the walking policy: |command| < 0.1 -> phase zeroed -> stand

# Command box of a command-conditioned planner's recording (vx m/s, vy m/s, wz rad/s): inside it the
# walking policy reaches 0.84-0.91 of vx, 0.69-0.74 of vy and 0.85-0.96 of wz with no fall; above
# 1.5 m/s forward the trunk sinks (0.364 m at 2.0 against 0.40) and the speed spread doubles.
CMD_LO = (-1.0, -0.75, -1.0)
CMD_HI = (1.5, 0.75, 1.0)


def foot_offsets(model) -> np.ndarray:
    """(4, 3) foot-sphere centre in its calf body's frame, read from the model."""
    import mujoco

    out = []
    for g, b in zip(FOOT_GEOMS, FOOT_BODIES):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        assert gid >= 0 and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) == b, (g, b)
        out.append(model.geom_pos[gid].copy())
    return np.stack(out)


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, u = q[..., :1], q[..., 1:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def foot_centres(body_pos_w: np.ndarray, body_quat_w: np.ndarray, body_names: list[str], offsets: np.ndarray) -> np.ndarray:
    """(T, 4, 3) world foot-sphere centres of a clip (FL, FR, RL, RR) from its calf body states."""
    idx = [body_names.index(b) for b in FOOT_BODIES]
    return body_pos_w[:, idx] + _quat_apply(body_quat_w[:, idx], np.broadcast_to(offsets, body_pos_w[:, idx].shape))


# Actual-velocity planners (``rover_v2_l1``): the planner's velocity input is the robot's actual
# velocity, capped at 1.5 m/s forward. The walking policy reaches 1.46 / 1.55 / 1.63 m/s at 1.6 / 1.7 /
# 1.8 commanded (1.49-1.52 at 1.7 with full sideways or turn), so the recording's box reaches 1.7
# forward; the other edges stay at the policy's trained box. Its ramps are capped at ACC_CMD_V2.
# JOY_* is the box in actual units these planners take, inside what the data reaches.
CMD_LO_V2 = (-1.0, -0.75, -1.0)
CMD_HI_V2 = (1.7, 0.75, 1.0)
ACC_CMD_V2 = (2.0, 2.0, 3.0)  # m/s^2, m/s^2, rad/s^2
JOY_LO = (-0.8, -0.5, -0.9)
JOY_HI = (1.5, 0.5, 0.9)


@dataclass
class CommandCfg:
    lo: tuple[float, float, float] = CMD_LO
    hi: tuple[float, float, float] = CMD_HI
    hold: tuple[float, float] = (1.0, 5.0)  # s a command is held
    ramp: tuple[float, float] = (0.2, 1.0)  # s of linear ramp to the next one
    p_stand: float = 0.15  # zero command
    p_turn: float = 0.15  # pure yaw rate
    p_axis: float = 0.25  # one translation axis (forward/back or sideways)
    min_norm: float = 0.15  # a non-zero held command is at least this far from zero in |(vx, vy, wz)|
    acc_max: tuple[float, float, float] | None = None  # per-axis cap on a ramp's rate (the ramp is lengthened); None: no cap


class RoverCommands:
    """Per-environment piecewise-linear velocity commands, numpy (asymmetric box, ``min_norm`` rule).

    Every environment starts standing for one hold, then repeatedly ramps over ``U(ramp)`` s to a
    new draw and holds it for ``U(hold)`` s. A draw is zero with ``p_stand``, a pure turn with
    ``p_turn``, a single translation axis with ``p_axis`` (forward/back or sideways, equally
    likely), otherwise uniform in the box; non-zero draws below ``min_norm`` are redrawn.
    """

    def __init__(self, n: int, cfg: CommandCfg | None = None, seed: int = 0):
        self.cfg = cfg or CommandCfg()
        self.n = n
        self.lo = np.asarray(self.cfg.lo, np.float64)
        self.hi = np.asarray(self.cfg.hi, np.float64)
        assert (self.lo <= 0).all() and (self.hi >= 0).all(), "the box must contain the zero command"
        self.rng = np.random.default_rng(seed)
        self.src = np.zeros((n, 3))
        self.dst = np.zeros((n, 3))
        self.cur = np.zeros((n, 3))
        self.t_switch = np.zeros(n)
        self.ramp_s = np.ones(n)
        self.t_next = self.rng.uniform(*self.cfg.hold, n)  # every env starts standing for one hold

    def _draw(self, k: int) -> np.ndarray:
        c = self.cfg
        box = self.rng.uniform(self.lo, self.hi, (k, 3))
        kind = self.rng.uniform(size=k)
        stand = kind < c.p_stand
        turn = (kind >= c.p_stand) & (kind < c.p_stand + c.p_turn)
        axis = (kind >= c.p_stand + c.p_turn) & (kind < c.p_stand + c.p_turn + c.p_axis)
        lateral = self.rng.uniform(size=k) < 0.5
        box[stand] = 0.0
        box[turn, 0:2] = 0.0
        box[axis, 2] = 0.0
        box[axis & lateral, 0] = 0.0
        box[axis & ~lateral, 1] = 0.0
        return box

    def sample(self, k: int) -> np.ndarray:
        out = self._draw(k)
        for _ in range(1000):
            norm = np.linalg.norm(out, axis=1)
            redo = (norm > 0) & (norm < self.cfg.min_norm)
            if not redo.any():
                return out
            out[redo] = self._draw(int(redo.sum()))
        raise RuntimeError("min_norm rejection did not converge; is the box smaller than min_norm?")

    def step(self, t: float) -> np.ndarray:
        """The commands in force at time ``t`` (s), (n, 3); call with non-decreasing ``t``."""
        due = np.flatnonzero(t >= self.t_next)
        if len(due):
            k = len(due)
            self.src[due] = self.cur[due]
            self.dst[due] = self.sample(k)
            self.t_switch[due] = t
            self.ramp_s[due] = self.rng.uniform(*self.cfg.ramp, k)
            if self.cfg.acc_max is not None:
                need = (np.abs(self.dst[due] - self.src[due]) / np.asarray(self.cfg.acc_max, np.float64)).max(1)
                self.ramp_s[due] = np.maximum(self.ramp_s[due], need)
            self.t_next[due] = t + self.ramp_s[due] + self.rng.uniform(*self.cfg.hold, k)
        w = np.clip((t - self.t_switch) / self.ramp_s, 0.0, 1.0)
        self.cur = self.src + (self.dst - self.src) * w[:, None]
        return self.cur

    def reset(self, ids, t: float) -> None:
        """Environments ``ids`` start standing again (after a fall)."""
        ids = np.asarray(ids)
        self.src[ids] = self.dst[ids] = self.cur[ids] = 0.0
        self.t_switch[ids] = t
        self.t_next[ids] = t + self.rng.uniform(*self.cfg.hold, len(ids))


# ---------------------------------------------------------------- the policy's gait clock
# The walking policy sees sin / cos of a free-running clock, (step * 0.02 s) mod 0.6 s, zeroed while
# |command| < 0.1. Which diagonal pair lifts first when a gait starts is set by that clock, which a
# planner conditioned on the command alone cannot see: from standing it has to guess the phase, and
# a deterministic model averages the guesses into shallow, sliding steps (slow turns in place slid
# 18-21 cm/s against 3 in the data). A planner may therefore take the clock as two more condition
# inputs; at deployment it runs its own clock.
POLICY_DT = 0.0025 * 8  # the walking policy's step (physics 2.5 ms x decimation 8)
PHASE_PERIOD = 0.6


def phase_condition(cmd: np.ndarray, clock_step: np.ndarray) -> np.ndarray:
    """(T, 2) the policy's phase input for commands (T, 3) applied at policy steps (T,)."""
    g = (np.asarray(clock_step, np.float64) * POLICY_DT) % PHASE_PERIOD / PHASE_PERIOD
    ph = np.stack([np.sin(2 * np.pi * g), np.cos(2 * np.pi * g)], -1)
    return np.where((np.linalg.norm(cmd, axis=-1) < PHASE_THRESHOLD)[:, None], 0.0, ph)


def planner_condition(cmd: np.ndarray, clock_step: np.ndarray | None, cond_dim: int) -> np.ndarray:
    """The condition a planner with ``cond_dim`` inputs takes: the command (3), plus the clock (5)."""
    if cond_dim == 3:
        return np.asarray(cmd)
    assert cond_dim == 5 and clock_step is not None
    return np.concatenate([cmd, phase_condition(cmd, clock_step)], -1)


def build_planner(cfg, normalizer, cond_dim: int = 3):
    """``planner.model.Planner`` with a ``cond_dim``-wide condition (the base model's is 3)."""
    from torch import nn

    from supermimic.planner.model import Planner

    model = Planner(cfg, normalizer)
    if cond_dim != 3:
        model.inp = nn.Linear(cfg.context * cfg.feat_dim + cond_dim + cfg.num_gaits, cfg.hidden)
    model.cond_dim = cond_dim
    return model


def load_planner(path, device="cpu"):
    """``Planner.load`` for checkpoints with any condition width (``cond_dim`` in the checkpoint, default 3)."""
    import torch

    from supermimic.planner.model import Normalizer, PlannerCfg

    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_planner(PlannerCfg(**ck["cfg"]), Normalizer(ck["mean"], ck["std"], ck["cond_scale"]), int(ck.get("cond_dim", 3))).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


def steady_mask(cmd: np.ndarray, settle_frames: int) -> np.ndarray:
    """Frames whose command equals that of each of the previous ``settle_frames`` frames."""
    changed = np.concatenate([[True], np.abs(np.diff(cmd, axis=0)).max(1) > 1e-6])
    last = np.maximum.accumulate(np.where(changed, np.arange(len(cmd)), 0))
    return np.arange(len(cmd)) - last >= settle_frames


# ---------------------------------------------------------------- actual-velocity planners
# These planners take the robot's actual velocity (heading-frame vx, vy, world yaw rate) in place of
# the policy's command, so the joystick says how fast the robot walks: a label per frame, either
# G(cmd) (``label_l1``, G = the policy's fitted steady-state map) or the centered one-period mean of
# the actual velocity (``label_l2``); scripts/planner/fit_velocity_map.py fits G and writes both. The
# condition keeps its shape (3 + the 2 clock inputs). The clock stays the policy's, gated on the
# policy's command; at deployment only the joystick value v exists, so the gate is evaluated on
# H(v) ~ G^-1(v), the command that walks at v. A checkpoint says which it is (``label``: "cmd"
# (command-conditioned, default), "l1", "l2") and carries the map and its joystick box;
# ``joystick_condition`` / ``joystick_cfg`` read both.


def monomial_exponents(degree: int) -> np.ndarray:
    """(M, 3) exponents of every monomial of (vx, vy, wz) of degree 1..``degree`` (no constant: G(0) = 0);
    the first three are vx, vy, wz. A fitted map stores its exponents, so the order only has to be fixed here."""
    import itertools

    return np.array([e for d in range(1, degree + 1) for e in itertools.product(range(d, -1, -1), repeat=3) if sum(e) == d])


@dataclass
class VelocityMap:
    """Polynomial map command -> steady actual velocity (``actual``) and its fitted inverse (``command``)."""

    exponents: np.ndarray  # (M, 3)
    fwd: np.ndarray  # (M, 3): actual = features(cmd) @ fwd
    inv: np.ndarray  # (M, 3): cmd = features(actual) @ inv

    @staticmethod
    def features(x: np.ndarray, exponents: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64)
        return np.prod(x[..., None, :] ** exponents, axis=-1)

    def actual(self, cmd: np.ndarray) -> np.ndarray:
        return self.features(cmd, self.exponents) @ self.fwd

    def command(self, v: np.ndarray) -> np.ndarray:
        return self.features(v, self.exponents) @ self.inv

    def to_dict(self) -> dict:
        return {"exponents": self.exponents.tolist(), "fwd": self.fwd.tolist(), "inv": self.inv.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> VelocityMap:
        return cls(np.asarray(d["exponents"], np.int64), np.asarray(d["fwd"], np.float64), np.asarray(d["inv"], np.float64))


def joystick_condition(ck: dict, v: np.ndarray, clock_step: np.ndarray | None) -> np.ndarray:
    """The condition of planner checkpoint ``ck`` for joystick values ``v`` (n, 3) in its own units
    (the policy's command for a command-conditioned planner, actual velocity otherwise) and its clock steps."""
    cond_dim = int(ck.get("cond_dim", 3))
    if ck.get("label", "cmd") == "cmd":
        return planner_condition(v, clock_step, cond_dim)
    v = np.asarray(v, np.float64)
    if cond_dim == 3:
        return v
    gate = VelocityMap.from_dict(ck["velocity_map"]).command(v)  # the policy command that walks at v
    return np.concatenate([v, phase_condition(gate, clock_step)], -1)


def joystick_cfg(ck: dict) -> CommandCfg:
    """The random joystick of the evaluation and the closed loop, in the checkpoint's units."""
    if ck.get("label", "cmd") == "cmd":
        return CommandCfg()
    return CommandCfg(lo=tuple(ck["joy_lo"]), hi=tuple(ck["joy_hi"]), acc_max=tuple(ck["joy_acc"]))


class RateLimiter:
    """The deployment stick, shaped like the recording's command stream (``RoverCommands``).

    Every change of the target starts a linear ramp from the current value to it. All axes ramp
    together over max(``min_ramp_s``, the longest axis at its ``acc`` cap) s, which is the recording's
    rule with its shortest ramp, 0.2 s; the recording drew U(0.2, 1.0) s. A target that changes again
    mid-ramp starts a new ramp from where the stick is, so a continuously moving stick is followed
    with a lag of about ``min_ramp_s``.

    A bare rate limit lets small changes finish in 0.1 s, faster than anything in the data, and
    ``rover_v2_l1`` then rocked in place after a slow turn was released.
    """

    def __init__(self, acc, dt: float = 0.02, n: int = 1, min_ramp_s: float = 0.2):
        self.acc = np.asarray(acc, np.float64)
        self.dt, self.min_ramp = float(dt), float(min_ramp_s)
        self.v = np.zeros((n, 3))
        self.src, self.dst = np.zeros((n, 3)), np.zeros((n, 3))
        self.ramp_s, self.elapsed = np.ones(n), np.zeros(n)

    def __call__(self, target: np.ndarray) -> np.ndarray:
        target = np.broadcast_to(np.asarray(target, np.float64), self.v.shape)
        new = np.abs(target - self.dst).max(1) > 1e-9
        if new.any():
            self.src[new], self.dst[new], self.elapsed[new] = self.v[new], target[new], 0.0
            self.ramp_s[new] = np.maximum(self.min_ramp, (np.abs(self.dst[new] - self.src[new]) / self.acc).max(1))
        self.elapsed += self.dt
        w = np.clip(self.elapsed / self.ramp_s, 0.0, 1.0)
        self.v = self.src + (self.dst - self.src) * w[:, None]
        return self.v.copy()
