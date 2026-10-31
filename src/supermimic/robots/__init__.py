"""Robot registry: ``get_robot_spec(name)``. Add a robot by writing ``robots/<name>.py`` with a
``<name>_spec(**overrides)`` function and one line in ``ROBOTS`` (docs/PORTING.md)."""

from __future__ import annotations

import importlib

from supermimic.robots.spec import DomainRandomization, ObsNoise, ResetNoise, RobotSpec, SimSettings, StateScales

__all__ = ["RobotSpec", "StateScales", "ObsNoise", "DomainRandomization", "ResetNoise", "SimSettings", "get_robot_spec", "set_robot_overrides", "get_robot_overrides", "ROBOT_NAMES"]

# name -> "module:function" returning the RobotSpec. g1 is the upstream humanoid (kept as the
# reference implementation and for the regression tests); rover is the Dobot Rover X1.
ROBOTS: dict[str, str] = {
    "g1": "supermimic.robots.g1:g1_spec",
    "rover": "supermimic.robots.rover:rover_spec",
}
ROBOT_NAMES = tuple(ROBOTS)
_CACHE: dict[str, RobotSpec] = {}
# Process-wide actuator overrides (PD stiffness, action scale), set once from the run config /
# checkpoint before the first ``get_robot_spec`` call so that every consumer (env actuator, torque
# model, world model) sees the same values. A spec function that does not take one rejects it.
_OVERRIDES: dict[str, float | None] = {"stiffness": None, "action_scale": None}


def set_robot_overrides(stiffness: float | None = None, action_scale: float | None = None) -> None:
    new = {"stiffness": stiffness, "action_scale": action_scale}
    if new != _OVERRIDES:
        _OVERRIDES.update(new)
        _CACHE.clear()


def get_robot_overrides() -> dict[str, float | None]:
    return dict(_OVERRIDES)


def get_robot_spec(name: str) -> RobotSpec:
    if name not in _CACHE:
        if name not in ROBOTS:
            raise KeyError(f"unknown robot {name!r}; known: {ROBOT_NAMES}")
        module, fn = ROBOTS[name].split(":")
        kwargs = {k: v for k, v in _OVERRIDES.items() if v is not None}
        try:
            _CACHE[name] = getattr(importlib.import_module(module), fn)(**kwargs)
        except TypeError as e:
            raise ValueError(f"robot {name!r} does not support the overrides {sorted(kwargs)}") from e
    return _CACHE[name]
