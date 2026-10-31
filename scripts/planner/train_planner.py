"""Train the walking planner on walking clips (docs/PLANNER.md).

Modules: features (``planner/features.py``, 20 per frame for 12 joints), dataset
(``planner/dataset.py``, split by recording episode: ``_e<k>_`` with k % 10 == 9 is validation),
model (``planner/model.py``: residual MLP on 25 context frames + the condition, predicting 50 frames
as a residual on the last context frame; 1024 x 4 blocks, ~10 M parameters). Condition = the stick
value at the first planned frame; after ``ar_warmup`` steps each batch also replans
R ~ U{0..ar_max} times every ``replan`` frames from its own detached output (2 s at ar_max 20), each
plan scored against the ground truth of its own window. Loss: MSE of normalised features +
``vel_weight`` x MSE of their frame-to-frame differences (including the step from the last context
frame into the plan).

``--phase`` adds the walking policy's gait clock to the condition (sin / cos of its free-running
0.6 s clock, zeroed below |command| 0.1 as the policy has it: ``planner.rover.phase_condition``,
from the clips' ``clock_step``), 5 inputs instead of 3; at deployment the planner runs its own clock.

``--label l1`` / ``l2`` (actual-velocity planners): the velocity inputs are the clips' ``label_l1``
(G(cmd), the policy's steady-state map) or ``label_l2`` (centered one-period mean of the actual
velocity) written by ``fit_velocity_map.py``, instead of the command; the clock stays gated on the
policy's command. The checkpoint carries ``label``, the map and the joystick box
(``planner.rover.joystick_condition`` / ``joystick_cfg`` read them).

The defaults are ``rover_v2_l1``'s (12.5 h of walking, 20k steps, 26 min on one RTX 5070):

    uv run python scripts/planner/train_planner.py --npz-dir data/planner/rover_v2/npz
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.planner.dataset import load_gait, to_split
from supermimic.planner.features import feat_slices
from supermimic.planner.model import Normalizer, PlannerCfg
from supermimic.planner.rover import FEAT_DIM, HOME_VEC, NJ, build_planner, planner_condition

ROOT = Path(__file__).resolve().parents[2]


def main(npz_dir: Path | None = None, run: str = "rover_v2_l1", steps: int = 20000, batch: int = 1024, lr: float = 3e-4, weight_decay: float = 1e-4, context: int = 25,
         horizon: int = 50, hidden: int = 1024, blocks: int = 4, dropout: float = 0.0, replan: int = 5, ar_max: int = 20, ar_warmup: int = 5000, vel_weight: float = 1.0,
         cond_source: str = "cmd", cond_frame: int | None = None, val_every: int = 10, device: str = "cuda:0", seed: int = 0, log_every: int = 500, eval_every: int = 2500,
         phase: bool = True, label: str = "l1") -> None:
    torch.manual_seed(seed)
    npz_dir = npz_dir or ROOT / "data" / "planner" / "rover_v2" / "npz"
    out = ROOT / "logs" / "planner" / run
    out.mkdir(parents=True, exist_ok=True)
    home = HOME_VEC.copy()
    t0 = time.time()
    tr, va = load_gait(npz_dir, home, 0, val_every, cond_source)
    cond_dim = 5 if phase else 3
    assert label in ("cmd", "l1", "l2")
    if phase or label != "cmd":
        assert cond_source == "cmd", "the clock and the labels go with the command condition"

        def with_clock(clips):
            out = []
            for feat, cond, g, name in clips:
                with np.load(npz_dir / f"{name}.npz") as z:
                    c = planner_condition(cond, z["clock_step"], cond_dim) if phase else cond  # the clock is gated on the policy's command
                    if label != "cmd":
                        c = np.concatenate([np.asarray(z[f"label_{label}"], np.float64), c[:, 3:]], -1)
                    out.append((feat, c, g, name))
            return out

        tr, va = with_clock(tr), with_clock(va)
    train, val = to_split(tr, device), to_split(va, device)
    print(f"[planner] rover: train {len(tr)} clips {train.feat.shape[0] / 50 / 3600:.2f} h, val {len(va)} clips {val.feat.shape[0] / 50 / 3600:.2f} h ({time.time() - t0:.0f} s to load)", flush=True)

    mean = train.feat.mean(0)
    std = train.feat.std(0).clamp_min(1e-3)
    cond_scale = train.cond.abs().amax(0).clamp_min(1e-3)
    cfg = PlannerCfg(feat_dim=FEAT_DIM, context=context, horizon=horizon, num_gaits=1, hidden=hidden, blocks=blocks, dropout=dropout)
    model = build_planner(cfg, Normalizer(mean, std, cond_scale), cond_dim).to(device)
    print(f"[planner] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params; cond scale {cond_scale.tolist()}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    span = context + ar_max * replan + horizon
    starts_tr = train.window_starts(span)
    starts_va = val.window_starts(span)
    g_eval = torch.Generator(device=device).manual_seed(1234)
    eval_idx = starts_va[torch.randint(len(starts_va), (4096,), device=device, generator=g_eval)]
    cf = cond_frame if cond_frame is not None else (0 if cond_source == "cmd" else horizon - 1)
    sl = feat_slices(NJ)
    offs = torch.arange(span, device=device)

    def run_batch(split, s, R):
        x = model.normalizer.norm(split.feat[s[:, None] + offs])  # (B, span, D)
        cond_all = split.cond[s[:, None] + offs]
        g = split.gait[s]
        stream = x[:, :context]
        losses, preds = [], []
        for r in range(R + 1):
            a = r * replan
            ctx = stream[:, -context:]
            target = x[:, a + context : a + context + horizon]
            pred = model(ctx, cond_all[:, a + context + cf], g)
            d_pred = torch.diff(torch.cat([ctx[:, -1:], pred], 1), dim=1)
            d_gt = torch.diff(torch.cat([x[:, a + context - 1 : a + context], target], 1), dim=1)
            losses.append(((pred - target) ** 2).mean() + vel_weight * ((d_pred - d_gt) ** 2).mean())
            preds.append((pred, target))
            stream = torch.cat([stream, pred[:, :replan].detach()], 1)
        return torch.stack(losses).mean(), preds

    @torch.no_grad()
    def evaluate():
        model.eval()
        res = {}
        for R in (0, ar_max):
            loss, preds = run_batch(val, eval_idx, R)
            res[f"val_loss_R{R}"] = float(loss)
            pred, target = preds[-1]
            p, t = model.normalizer.denorm(pred), model.normalizer.denorm(target)
            for name, k in (("joint_rad", "joint"), ("height_mm", "height"), ("lin_h_mps", "lin_h"), ("yaw_rate_rps", "yaw_rate")):
                err = (p[..., sl[k]] - t[..., sl[k]]).pow(2).mean(-1).sqrt()  # (B, T)
                scale = 1000.0 if k == "height" else 1.0
                res[f"R{R}_{name}_first10"] = float(err[:, :10].mean() * scale)
                res[f"R{R}_{name}_all"] = float(err.mean() * scale)
        model.train()
        return res

    log = (out / "train.log").open("w")
    best = float("inf")
    t0 = time.time()
    model.train()
    extra = {"gait": "rover", "robot": "rover", "cond_source": cond_source, "cond_frame": cf, "replan": replan, "home": home, "npz_dir": str(npz_dir), "cond_dim": cond_dim, "label": label}
    if label != "cmd":
        vmap = json.loads((npz_dir.parent / "velocity_map.json").read_text())
        extra.update(velocity_map={k: vmap[k] for k in ("exponents", "fwd", "inv")}, joy_lo=vmap["joy_lo"], joy_hi=vmap["joy_hi"], joy_acc=vmap["joy_acc"])
    for step in range(1, steps + 1):
        R = int(torch.randint(ar_max + 1, ())) if step > ar_warmup else 0
        s = starts_tr[torch.randint(len(starts_tr), (batch,), device=device)]
        loss, _ = run_batch(train, s, R)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0:
            msg = f"step {step} | {time.time() - t0:.0f} s | loss {float(loss):.4f} (R={R}) | lr {sched.get_last_lr()[0]:.2e}"
            print(msg, flush=True)
            log.write(msg + "\n")
        if step % eval_every == 0 or step == steps:
            res = evaluate()
            msg = f"eval {step} | " + " ".join(f"{k} {v:.4f}" for k, v in res.items())
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            score = res[f"val_loss_R{ar_max}"]
            if step > ar_warmup and score < best:
                best = score
                model.save(out / "planner.pt", step=step, eval=res, **extra)
            model.save(out / "planner_last.pt", step=step, eval=res, **extra)
    wall = time.time() - t0
    (out / "config.json").write_text(json.dumps({"robot": "rover", "npz_dir": str(npz_dir), "steps": steps, "batch": batch, "lr": lr, "weight_decay": weight_decay, "context": context,
                                                  "horizon": horizon, "hidden": hidden, "blocks": blocks, "dropout": dropout, "replan": replan, "ar_max": ar_max, "ar_warmup": ar_warmup,
                                                  "vel_weight": vel_weight, "cond_source": cond_source, "cond_dim": cond_dim, "label": label, "cond_frame": cf, "val_every": val_every, "seed": seed, "best_val_loss_ar": best,
                                                  "wall_s": round(wall, 1), "train_hours": train.feat.shape[0] / 50 / 3600, "val_hours": val.feat.shape[0] / 50 / 3600}, indent=2))
    print(f"[planner] done in {wall:.0f} s, best val loss (R={ar_max}) {best:.4f} -> {out}")


if __name__ == "__main__":
    tyro.cli(main)
