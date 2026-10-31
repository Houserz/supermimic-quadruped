"""Planner dataset: per-frame features and velocity conditions of a gait's clips, on the GPU.

Clips are walking rollouts in the mjlab npz layout plus a per-frame ``cmd`` (docs/PLANNER.md).
Each frame becomes the ``features.features_from_pose`` vector; each frame's condition is either the
command the recorded policy was given at that frame (``cond_source="cmd"``, the clip's ``cmd``) or
the achieved velocity (``"vel"``: ``features.heading_velocity``, the centered 25-frame mean
heading-frame velocity and yaw rate). Clips are concatenated; windows never cross a clip boundary. The split is by recording environment (``_e<env>_`` in the name), so
validation windows come from robots the training windows never saw.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from supermimic.planner.features import features_from_pose, heading_velocity


@dataclass
class Split:
    feat: torch.Tensor  # (N, D) concatenated frames
    cond: torch.Tensor  # (N, 3)
    gait: torch.Tensor  # (N,) gait index
    clip_start: torch.Tensor  # (N,) first frame of each frame's clip
    clip_end: torch.Tensor  # (N,) one past the last frame of each frame's clip
    names: list[str]

    def window_starts(self, span: int, margin: int = 12) -> torch.Tensor:
        """Every frame index s with s + span + margin <= its clip's end (margin: the condition's half window)."""
        idx = torch.arange(self.feat.shape[0], device=self.feat.device)
        return idx[idx + span + margin <= self.clip_end]


def load_gait(npz_dir: Path, home: np.ndarray, gait_index: int, val_every: int = 10, cond_source: str = "cmd") -> tuple[list, list]:
    """Per-clip (features, condition, gait, name) lists for train and validation (env % val_every == val_every - 1)."""
    train, val = [], []
    for f in sorted(npz_dir.glob("*.npz")):
        with np.load(f) as z:
            rp = np.asarray(z["body_pos_w"][:, 0], np.float64)
            rq = np.asarray(z["body_quat_w"][:, 0], np.float64)
            q = np.asarray(z["joint_pos"], np.float64)
            cmd = np.asarray(z["cmd"], np.float64) if cond_source == "cmd" else None
        feat = features_from_pose(rp, rq, q, home)
        cond = cmd if cond_source == "cmd" else heading_velocity(rp, rq)
        m = re.search(r"_e(\d+)_", f.stem)
        env = int(m.group(1)) if m else 0
        (val if env % val_every == val_every - 1 else train).append((feat, cond, gait_index, f.stem))
    return train, val


def to_split(clips: list, device) -> Split:
    feats, conds, gaits, starts, ends, names = [], [], [], [], [], []
    n = 0
    for feat, cond, g, name in clips:
        T = len(feat)
        feats.append(feat)
        conds.append(cond)
        gaits.append(np.full(T, g))
        starts.append(np.full(T, n))
        ends.append(np.full(T, n + T))
        names.append(name)
        n += T
    cat = lambda xs, dt: torch.as_tensor(np.concatenate(xs), dtype=dt, device=device)
    return Split(cat(feats, torch.float32), cat(conds, torch.float32), cat(gaits, torch.long), cat(starts, torch.long), cat(ends, torch.long), names)
