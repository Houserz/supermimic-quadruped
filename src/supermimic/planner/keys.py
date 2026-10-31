"""Keyboard joystick for the MuJoCo viewer: held keys -> stick axes, plus one-shot events.

Keys (hold to move, release to stop):

  Up / Down              walk forward / backward
  Left / Right           turn left / right
  Shift + Left / Right   walk sideways left / right
  keypad 8 2 4 6         forward / backward / turn left / turn right; keypad 7 9 sideways; keypad 5 stop
  Page Up / Page Down    speed level up / down (25 / 50 / 75 / 100 % of the planner's box)
  Backspace              reset the robot, standing
  F6 / keypad 1          skill: sit                F7 / keypad 3          skill: shake hands
  F8 / keypad 0          skill: high five          F9 / keypad /          skill: stretch
  F10 / keypad *         skill: dance (a pose medley)

Every letter is a MuJoCo viewer shortcut (W = wireframe, S = shadow, ...), so none is used. The
viewer's ``key_callback`` reports presses only; a GLFW key callback is put in front of MuJoCo's on
the viewer window to see releases too. Without GLFW each press holds its key for 0.4 s.
"""

from __future__ import annotations

import ctypes
import threading
import time

import numpy as np

# GLFW key codes and actions.
UP, DOWN, LEFT, RIGHT, PAGE_UP, PAGE_DOWN, BACKSPACE = 265, 264, 263, 262, 266, 267, 259
KP = {str(i): 320 + i for i in range(10)}
KP_DIVIDE, KP_MULTIPLY = 331, 332
F6, F7, F8, F9, F10 = 295, 296, 297, 298, 299
RELEASE, PRESS = 0, 1
MOD_SHIFT = 0x0001
LEVELS = (0.25, 0.5, 0.75, 1.0)
SKILL_KEYS = {F6: "sit", KP["1"]: "sit", F7: "shake", KP["3"]: "shake", F8: "high_five", KP["0"]: "high_five",
              F9: "stretch", KP_DIVIDE: "stretch", F10: "dance", KP_MULTIPLY: "dance"}


class Keys:
    """Held keys (with the mode a Left/Right press started in) and one-shot events, shared with the viewer thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.held: dict[int, str] = {}
        self.events: list[str] = []
        self.hooked = False
        self.tap_until: dict[int, float] = {}

    def edge(self, key: int, down: bool, mods: int) -> None:
        with self.lock:
            if key in SKILL_KEYS:
                if down:
                    self.events.append("skill:" + SKILL_KEYS[key])
                return
            if down:
                if key in (PAGE_UP, PAGE_DOWN, BACKSPACE, KP["5"]):
                    self.events.append({PAGE_UP: "faster", PAGE_DOWN: "slower", BACKSPACE: "reset", KP["5"]: "stop"}[key])
                    return
                self.held[key] = "strafe" if (key in (LEFT, RIGHT) and mods & MOD_SHIFT) else "move"
            else:
                self.held.pop(key, None)

    def tap(self, key: int) -> None:
        """No release events available: a press holds the key for 0.4 s."""
        self.edge(key, True, 0)
        with self.lock:
            if key in self.held:
                self.tap_until[key] = time.monotonic() + 0.4

    def set_held(self, held: dict[int, str]) -> None:
        """Replace the held keys (scripted driving)."""
        with self.lock:
            self.held = dict(held)

    def axes(self) -> tuple[float, float, float]:
        """(forward, sideways, turn), each in {-1, 0, 1}."""
        with self.lock:
            now = time.monotonic()
            for k, t in list(self.tap_until.items()):
                if now > t:
                    self.held.pop(k, None)
                    self.tap_until.pop(k)
            h = dict(self.held)
        fwd = float(UP in h or KP["8"] in h) - float(DOWN in h or KP["2"] in h)
        side = float(h.get(LEFT) == "strafe" or KP["7"] in h) - float(h.get(RIGHT) == "strafe" or KP["9"] in h)
        turn = float(h.get(LEFT) == "move" or KP["4"] in h) - float(h.get(RIGHT) == "move" or KP["6"] in h)
        return fwd, side, turn

    def lit(self) -> set[str]:
        """The arrow-pad keys to show pressed: "up", "down", "left", "right", "shift" (keypad keys map onto them)."""
        fwd, side, turn = self.axes()
        out = {n for n, on in (("up", fwd > 0), ("down", fwd < 0), ("left", side > 0 or turn > 0), ("right", side < 0 or turn < 0)) if on}
        if side:
            out.add("shift")
        return out

    def pop_events(self) -> list[str]:
        with self.lock:
            ev, self.events = self.events, []
        return ev


KEYS = Keys()


def stick_target(keys: Keys, level: int, lo, hi) -> np.ndarray:
    """The keys' stick value at a speed level, inside the planner's asymmetric joystick box."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    axes = np.array(keys.axes())
    return np.where(axes >= 0, axes * hi, -axes * lo) * LEVELS[level]


# ---------------------------------------------------------------- GLFW hook (key releases)
_KEYFUN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
_previous: dict[int, object] = {}


def _on_glfw_key(window, key, scancode, action, mods) -> None:
    prev = _previous.get(window)
    if prev is not None:
        prev(window, key, scancode, action, mods)  # MuJoCo's own callback first, for its shortcuts
    if action == PRESS:
        KEYS.edge(key, True, mods)
    elif action == RELEASE:
        KEYS.edge(key, False, mods)


_ON_GLFW_KEY = _KEYFUN(_on_glfw_key)  # kept alive: GLFW holds a raw pointer to it
_ON_GLFW_KEY_ADDR = ctypes.cast(_ON_GLFW_KEY, ctypes.c_void_p).value


def key_callback(key: int) -> None:
    """``launch_passive``'s callback (presses only, on the viewer thread): install the hook on the first press."""
    if KEYS.hooked:
        return  # the hook reports this press itself
    try:
        import glfw

        lib = glfw._glfw
        current = ctypes.CFUNCTYPE(ctypes.c_void_p)(("glfwGetCurrentContext", lib))
        set_cb = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("glfwSetKeyCallback", lib))
        window = current()
    except Exception as e:  # noqa: BLE001
        window = None
        print(f"[teleop] GLFW not reachable ({e}): each press holds its key for 0.4 s")
    if not window:
        KEYS.tap(key)
        return
    replaced = set_cb(window, _ON_GLFW_KEY_ADDR)
    _previous[window] = _KEYFUN(replaced) if replaced else None
    KEYS.hooked = True
    print("[teleop] key hook installed: holding a key moves, releasing it stops", flush=True)
    KEYS.edge(key, True, 0)  # this first press; its release comes through the hook
