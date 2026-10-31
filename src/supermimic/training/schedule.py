"""Cosine learning-rate schedule (paper Table A.5): decay to `final_ratio` over `decay_fraction` of training, then constant."""

from __future__ import annotations

import math


def cosine_lr(
    step: int, total_steps: int, base_lr: float, final_ratio: float = 0.03, decay_fraction: float = 250_000 / 320_000, start: int = 0
) -> float:
    """lr at `step`: base_lr until `start`, cosine from `start` to `start + (total_steps - start) * decay_fraction`,
    then base_lr * final_ratio. `start` > 0 anneals a resumed run whose first phase ran at a (nearly) constant lr."""
    step = step - start
    decay_steps = max(1, int((total_steps - start) * decay_fraction))
    if step <= 0:
        return base_lr
    if step >= decay_steps:
        return base_lr * final_ratio
    cos = 0.5 * (1.0 + math.cos(math.pi * step / decay_steps))
    return base_lr * (final_ratio + (1.0 - final_ratio) * cos)
