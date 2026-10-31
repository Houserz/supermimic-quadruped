"""World model: learned dynamics + learned termination head + analytic observation/reward heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.world_model.dynamics import MLP, DynamicsMLP
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

# Termination logit of an immune episode (plan D22): sigmoid(-30) ~ 1e-13, and no gradient reaches the head.
IMMUNE_LOGIT = -30.0


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8
    support_foot_threshold: float = 0.0  # > 0 enables the third termination feature (support feet)


class TerminationHead(nn.Module):
    """P(terminal | state, reference). Input: analytic termination features and their magnitudes."""

    def __init__(self, hidden=(64, 64), height_thr: float = 0.25, tilt_thr: float = 0.8, foot_thr: float | None = None):
        super().__init__()
        scales = [1.0 / height_thr, 1.0 / tilt_thr] + ([1.0 / foot_thr] if foot_thr else [])
        self.n_feat = len(scales)
        self.mlp = MLP(2 * self.n_feat, tuple(hidden), 1)
        self.register_buffer("scale", torch.tensor(scales))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (…, n_feat) = [height error, tilt error (, foot error)] -> logits (…,)"""
        x = feats * self.scale
        return self.mlp(torch.cat([x, x.abs()], dim=-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg.state_spec, device)
        nj = self.spec.nj
        self.dynamics = DynamicsMLP(feat_dim=self.spec.feat_dim, act_dim=nj, xi_dim=self.spec.xi_dim, hidden=cfg.hidden, out_dim=self.spec.delta_dim)
        foot_thr = cfg.support_foot_threshold if cfg.support_foot_threshold > 0 else None
        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold, foot_thr)
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        n_head_feat = getattr(heads, "num_termination_features", 2)
        assert n_head_feat == self.term_head.n_feat, f"termination features: heads give {n_head_feat}, head expects {self.term_head.n_feat}"
        self.to(device)

    # ------------------------------------------------------------------ one step

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi))."""
        feat = self.spec.canonical(s)
        delta = self.dynamics(feat, action, self.spec.normalize_xi(xi))
        return self.spec.integrate(s, delta)

    def _term_feats(self, s_next: torch.Tensor, ref: ReferenceFrame, bodies=None) -> torch.Tensor:
        fn = getattr(self.heads, "termination_features_of", None)
        if fn is None:  # bare heads (tests): root pose only
            return self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ref)
        return fn(s_next, ref, bodies)

    def termination_logits(self, s_next: torch.Tensor, ref: ReferenceFrame, immune: torch.Tensor | None = None, bodies=None) -> torch.Tensor:
        """immune (…,) bool: the episode's root terminations are disabled, so it cannot terminate."""
        logits = self.term_head(self._term_feats(s_next, ref, bodies))
        return logits if immune is None else logits.masked_fill(immune, IMMUNE_LOGIT)

    # ------------------------------------------------------------------ Eq. 1

    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat, ref_body_pos, live):
        """One autoregressive step of Eq. 1: returns (s_next, sq_err (n,D), term_bce (n,), logits (n,)).

        live (n,) float: 0 for immune episodes, whose labels say nothing about the termination rule.
        ref_body_pos: (n, B, 3) tracked-body reference positions (only read for the support-foot feature).
        """
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        ref = ReferenceFrame(ref_body_pos, None, None, None, ref_root_pos, ref_root_quat)
        logits = self.term_head(self._term_feats(s_next, ref))
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none") * live
        return s_next, e * e, bce, logits

    def enable_compile(self, mode: str | None = "default") -> None:
        self._loss_step_fn = torch.compile(self._loss_step, mode=None if mode == "default" else mode, dynamic=False) if mode else self._loss_step

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, done: torch.Tensor, refs: list[ReferenceFrame], immune: torch.Tensor | None = None):
        """k-step autoregressive loss.

        states: (k+1, n, S) logged states; a_exec: (k, n, A); xi: (n, X);
        done: (k, n) termination label of states 1..k; refs[j]: reference of transition j (for the
        termination features of state j+1); immune: (n,) bool, excluded from the termination loss.
        """
        k = a_exec.shape[0]
        s = states[0]
        live = torch.ones_like(done[0]) if immune is None else (~immune).to(done.dtype)
        n_live = live.sum().clamp_min(1.0)
        total_state = 0.0
        total_term = 0.0
        metrics = {}
        err_first = None
        err_last = None
        acc_num = 0.0
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        needs_feet = self.term_head.n_feat > 2
        for j in range(k):
            r = refs[j]
            body_pos = r.body_pos_w if needs_feet else torch.zeros(0, device=s.device, dtype=s.dtype)
            s, sq, bce, logits = step_fn(s, a_exec[j], xi, states[j + 1], done[j], r.root_pos_w, r.root_quat_w, body_pos, live)
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.sum() / n_live
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
                acc_num += (((logits > 0).float() == done[j]).float() * live).sum() / n_live
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        # Metrics stay on-device (detached tensors): a float() here would stall the CPU on the
        # forward pass every minibatch, which in data-parallel runs turns into lockstep waiting.
        metrics.update(
            state_loss=(total_state / k).detach(),
            term_bce=(total_term / k).detach(),
            nse_1step=err_first.detach(),
            nse_kstep=err_last.detach(),
            term_acc=(acc_num / k).detach(),
        )
        return loss, metrics

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), averaged over the batch."""
        k = a_exec.shape[0]
        s = states[0]
        out = []
        for j in range(k):
            s = self.step(s, a_exec[j], xi)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)
