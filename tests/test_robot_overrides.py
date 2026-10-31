"""Actuator overrides (kp) reach the actuator and leave the defaults alone."""

import pytest

from supermimic.robots import get_robot_spec, set_robot_overrides


@pytest.fixture(autouse=True)
def _reset_overrides():
    yield
    set_robot_overrides()


def test_rover_default_gains():
    set_robot_overrides()
    act = get_robot_spec("rover").entity_cfg_fn(False).articulation.actuators[0]
    assert act.stiffness == 25.0 and act.damping == pytest.approx(1.3)


def test_stiffness_override_reaches_actuator_and_clears():
    from supermimic.robots.rover import damping_for

    set_robot_overrides(stiffness=120.0)
    act = get_robot_spec("rover").entity_cfg_fn(False).articulation.actuators[0]
    assert act.stiffness == 120.0 and act.damping == pytest.approx(damping_for(120.0))
    set_robot_overrides()  # clearing restores the cached default
    assert get_robot_spec("rover").entity_cfg_fn(False).articulation.actuators[0].stiffness == 25.0


def test_unsupported_override_is_rejected():
    set_robot_overrides(action_scale=0.5)
    with pytest.raises(ValueError):
        get_robot_spec("rover")
