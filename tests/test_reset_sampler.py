"""Return-based reset sampler (plan D14) bookkeeping under get-up training (plan D22)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand


def test_immune_and_pushed_episodes_do_not_update_bins():
    # A fall in an immune or pushed episode leaves G_b untouched; an ordinary fall still counts.
    n, steps, max_steps = 3, 10, 64
    cmd = object.__new__(MultiClipMotionCommand)
    lib = SimpleNamespace(num_bins=4, bin_of_frame=lambda f: f // 50)  # one 200-frame clip, 1 s bins
    reward_buf = torch.zeros(n)
    cmd.__dict__.update(
        cfg=SimpleNamespace(gamma=0.99, ema_alpha=1.0, min_remaining_steps=100, sampler_skip_immune=True),
        _env=SimpleNamespace(num_envs=n, device="cpu", reward_buf=reward_buf, termination_manager=SimpleNamespace(terminated=torch.ones(n, dtype=torch.bool))),
        library=lib,
        frame=torch.tensor([0, 100, 150]),
        clip_end=torch.zeros(n, dtype=torch.bool),
        immune=torch.tensor([False, True, False]),
        pushed=torch.tensor([False, False, True]),
        bin_return=torch.full((4,), 10.0),
        max_ep_steps=max_steps,
        ep_rewards=torch.zeros(n, max_steps),
        ep_bins=torch.zeros(n, max_steps, dtype=torch.long),
        ep_len=torch.zeros(n, dtype=torch.long),
        _gamma_pow=0.99 ** torch.arange(max_steps, dtype=torch.float64),
        bootstrap_fn=None,
        metrics={"episode_return_to_go": torch.zeros(n)},
    )
    for _ in range(steps):  # every env earns little reward, then falls (terminated)
        reward_buf[:] = 0.01
        cmd._log_step_rewards(None)
        cmd.frame += 1
    cmd._finalize_episodes(torch.arange(n))
    assert cmd.bin_return[0] < 1.0  # the ordinary fall marks its bin hard
    assert cmd.bin_return[2] == pytest.approx(10.0) and cmd.bin_return[3] == pytest.approx(10.0)
    assert (cmd.ep_len == 0).all()
