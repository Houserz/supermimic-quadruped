"""Minimal BVH reader: hierarchy, motion channels and forward kinematics to world positions.

Conventions (checked against both dog sources in ``scripts/inspect_bvh.py``):

* A joint with position channels takes its local translation **from the channels**, not from
  its OFFSET (Blender's importer does the same); a joint without them uses the OFFSET. The
  Tencent skeleton has six channels on every joint, so this matters there.
* Rotation channels compose in the order listed: ``Zrotation Xrotation Yrotation`` gives
  ``R = Rz @ Rx @ Ry`` and a child sits at ``R_parent @ t_child + p_parent``.
* End sites are kept as joints named ``<parent>_end`` with no channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Joint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str] = field(default_factory=list)
    channel_start: int = 0


@dataclass
class Bvh:
    joints: list[Joint]
    frames: np.ndarray  # (T, n_channels)
    frame_time: float

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.joints]

    def index(self, name: str) -> int:
        return self.names.index(name)


def read(path: str | Path) -> Bvh:
    tokens = Path(path).read_text(errors="replace").split()
    i = 0
    joints: list[Joint] = []
    stack: list[int] = []
    n_ch = 0
    while tokens[i] != "MOTION":
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            joints.append(Joint(tokens[i + 1], stack[-1] if stack else -1, np.zeros(3)))
            i += 2
        elif t == "End":
            parent = stack[-1]
            joints.append(Joint(f"{joints[parent].name}_end", parent, np.zeros(3)))
            i += 2
        elif t == "{":
            stack.append(len(joints) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            joints[stack[-1]].offset = np.array([float(x) for x in tokens[i + 1 : i + 4]])
            i += 4
        elif t == "CHANNELS":
            k = int(tokens[i + 1])
            j = joints[stack[-1]]
            j.channels = tokens[i + 2 : i + 2 + k]
            j.channel_start = n_ch
            n_ch += k
            i += 2 + k
        else:
            i += 1
    i += 1
    assert tokens[i] == "Frames:", tokens[i]
    n_frames = int(tokens[i + 1])
    assert tokens[i + 2] == "Frame" and tokens[i + 3] == "Time:"
    frame_time = float(tokens[i + 4])
    values = np.array(tokens[i + 5 : i + 5 + n_frames * n_ch], dtype=np.float64)
    assert values.size == n_frames * n_ch, f"{path}: {values.size} values for {n_frames} x {n_ch}"
    return Bvh(joints, values.reshape(n_frames, n_ch), frame_time)


def forward_kinematics(b: Bvh) -> tuple[np.ndarray, np.ndarray]:
    """World positions (T, J, 3) and world rotation matrices (T, J, 3, 3), in the file's units/axes."""
    T = b.frames.shape[0]
    J = len(b.joints)
    pos = np.zeros((T, J, 3))
    rot = np.zeros((T, J, 3, 3))
    for j, joint in enumerate(b.joints):
        local_t = np.broadcast_to(joint.offset, (T, 3)).copy()
        local_r = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()
        rot_axes, rot_cols = [], []
        for c, ch in enumerate(joint.channels):
            col = joint.channel_start + c
            if ch.endswith("position"):
                local_t[:, "XYZ".index(ch[0])] = b.frames[:, col]
            elif ch.endswith("rotation"):
                rot_axes.append(ch[0].lower())
                rot_cols.append(col)
        if rot_axes:
            local_r = Rotation.from_euler("".join(rot_axes).upper(), b.frames[:, rot_cols], degrees=True).as_matrix()
        if joint.parent < 0:
            pos[:, j] = local_t
            rot[:, j] = local_r
        else:
            p = joint.parent
            pos[:, j] = pos[:, p] + np.einsum("tij,tj->ti", rot[:, p], local_t)
            rot[:, j] = rot[:, p] @ local_r
    return pos, rot
