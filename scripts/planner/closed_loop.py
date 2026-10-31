"""The walking planner drives the Rover's SuperMimic tracker in simulation (closed loop), with metrics.

Each environment starts standing at frame ``--start`` of a walking clip (the reset puts the robot in
that frame's reference state). Two goal sources, same starts:

* ``plan``: a random joystick (``RoverCommands`` in the planner's box, ``joystick_cfg``) drives the
  planner, which replans every 5 control steps from its own executed plan (its first context is the
  clip's preceding frames) and crossfades ``--blend`` frames; the tracker's goal is the plan's next
  10 frames (``planner.goal.plan_goal_frames``). Nothing in this loop reads the robot's state: the
  planner sees only its own plan, the stick and (a clock-conditioned planner, ``cond_dim`` 5) its own
  gait clock, started at the clip's clock and counting on.
* ``data``: the tracker follows the recorded walking itself (the clip's own goal), with the stick
  value the recording corresponds to (the command, or G(cmd) for an actual-velocity planner).

The starts are validation clips of the planner's walking data (``npz_dir`` in the checkpoint). Without
that data (the released checkpoint), every environment starts from the run's ``stand_seed.npz`` and
only ``plan`` runs. Every gain below is robot (goal) over the stick, ~1 when the robot walks at what
the stick says.

The clip only seeds the reset: reference terminations are off (height / tilt thresholds 1e3), a fall
is detected here (tilt above ``--fall-tilt`` or root below ``--fall-height``) and the frames after it
are excluded. The robot's truth (root pose) is used for the measurement only.

Metrics per source: falls; steady-state |actual - stick| heading velocity (centered 25-frame mean,
stick constant for ``--settle`` s); the gain achieved / commanded per axis (least-squares slope on
steady frames where that axis is commanded at least 0.2), split into pure turns, pure translations
and the rest, with the goal's own gain next to it, so the tracker's share is visible; tracking error
against the goal's current frame (joints, root height, tilt); torque saturation (share of steps with
a joint at its effort limit, 23 / 23 / 55 N*m) and peak |tau| / limit. Writes ``metrics.json`` and the
robot's trajectories (``<source>_robot_<k>.npz`` with ``cmd``).

    uv run python scripts/planner/closed_loop.py                       # rover_v2_l1 + rover_r2, 60 s x 25 robots
    uv run python scripts/planner/closed_loop.py --clips 125 --dr      # with domain randomisation
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.data.clips import body_states_from_pose
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.planner.features import features_from_pose, heading_velocity
from supermimic.planner.goal import plan_goal_frames
from supermimic.planner.rover import (
    HOME_VEC,
    NJ,
    RoverCommands,
    joystick_cfg,
    joystick_condition,
    load_planner,
    steady_mask,
)
from supermimic.robots import get_robot_spec
from supermimic.robots.rover import EFFORT_HIP, EFFORT_KNEE, JOINT_NAMES

ROOT = Path(__file__).resolve().parents[2]
GAIN_MIN_CMD = 0.2
# goal_features layout (motion_library.goal_features): q - q0 | qd | grav 3 | ang_b 3 | lin_h 3 | height 1
G_JOINT = slice(0, NJ)
G_GRAV = slice(2 * NJ, 2 * NJ + 3)
G_LIN_H = slice(2 * NJ + 6, 2 * NJ + 9)
G_HEIGHT = 2 * NJ + 9
G_DIM = 2 * NJ + 10


@torch.no_grad()
def run(source: str, policy, planner, ck_planner, files: list[Path], start: int, seconds: float, blend: int, dr: bool, device: str, seed: int):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.tracking_env_cfg import make_env_cfg

    n = len(files)  # one environment per entry; entries may repeat
    args = SuperMimicEnvCfgArgs(npz_files=sorted({str(f) for f in files}), robot="rover", num_envs=n, future_frames=10, episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=dr, reset_noise=False,
                                domain_randomization=dr, sampling_mode="start", record_transitions=False, seed=seed, root_height_threshold=1.0e3, root_tilt_threshold=1.0e3, support_foot_threshold=0.0)
    cfg = make_env_cfg(args)
    cfg.terminations.pop("clip_end")  # the clip only seeds the reset (the standing seed is 2 s long)
    configure_torch_backends()
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        cmd_term = env.command_manager.get_term("motion")
        lib = cmd_term.library
        clip_of = {c.name: i for i, c in enumerate(lib.clips)}
        idx = torch.tensor([clip_of[f.stem] for f in files], device=device)
        cmd_term.forced_frame[:] = lib.clip_start[idx] + start
        obs, _ = env.reset()
        robot = env.scene["robot"]
        origins = env.scene.env_origins
        home = torch.tensor(HOME_VEC, dtype=torch.float32, device=device)
        assert torch.allclose(home, lib.default_joint_pos, atol=1e-5), "planner HOME and the entity's default pose differ"
        steps = round(seconds * 50)
        K = 10
        log = {k: torch.zeros(steps, n, d, device=device) for k, d in (("root_pos", 3), ("root_quat", 4), ("joint", NJ), ("tau", NJ), ("goal0", G_DIM), ("cmd", 3))}

        if source == "plan":
            C = planner.cfg.context
            replan = int(ck_planner["replan"])
            ctx, clock0 = [], []
            for f in files:
                with np.load(f) as z:
                    clock0.append(int(z["clock_step"][start]) if "clock_step" in z.files else 0)  # the first planned frame's clock
                    ctx.append(features_from_pose(z["body_pos_w"][:, 0].astype(float), z["body_quat_w"][:, 0].astype(float), z["joint_pos"].astype(float), HOME_VEC)[start - C : start])
            stream = torch.tensor(np.stack(ctx), dtype=torch.float32, device=device)
            cmds = RoverCommands(n, joystick_cfg(ck_planner), seed=seed)  # in the planner's units (actual velocity for rover_v2_l1)
            gait = torch.zeros(n, dtype=torch.long, device=device)
            buf, G, i0 = None, None, 0
        else:
            ref_cmd = []
            for f in files:
                with np.load(f) as z:
                    # the stick equivalent of the recorded walking: the command, or for an actual-velocity planner G(cmd)
                    ref_cmd.append(np.asarray(z["cmd"] if ck_planner.get("label", "cmd") == "cmd" else z["label_l1"], np.float32)[start : start + steps])
            ref_cmd = torch.tensor(np.stack(ref_cmd, 1), device=device)  # (steps, n, 3)

        for t in range(steps):
            if source == "plan":
                if t % replan == 0:
                    c3 = cmds.step(t / 50.0).copy()
                    c = torch.as_tensor(c3, dtype=torch.float32, device=device)
                    cond = torch.as_tensor(joystick_condition(ck_planner, c3, np.array(clock0) + t), dtype=torch.float32, device=device)
                    new = planner.plan(stream[:, -C:], cond, gait)
                    if buf is not None and blend > 0:
                        w = torch.arange(1, blend + 1, device=device, dtype=torch.float32)[None, :, None] / (blend + 1)
                        new[:, :blend] = (1 - w) * buf[:, :blend] + w * new[:, :blend]
                    stream = torch.cat([stream, new[:, :replan]], 1)
                    buf = new[:, replan:]
                    G = plan_goal_frames(new, home, lib.default_joint_pos)
                    i0 = t
                goal = G[:, t - i0 : t - i0 + K].reshape(n, -1)
                log["cmd"][t] = c
            else:
                goal = obs["goal"]
                log["cmd"][t] = ref_cmd[t]
            log["goal0"][t] = goal[:, :G_DIM]
            act = policy(obs["policy"], goal)
            obs, _, _, _, _ = env.step(act)
            log["root_pos"][t] = robot.data.root_link_pos_w - origins
            log["root_quat"][t] = robot.data.root_link_quat_w
            log["joint"][t] = robot.data.joint_pos
            log["tau"][t] = robot.data.qfrc_actuator  # joint order (actuator_force is in actuator order: hip group, then knees)
        L = {k: v.cpu().numpy() for k, v in log.items()}
    finally:
        env.close()
    return L


def falls(L: dict, fall_tilt: float, fall_height: float) -> np.ndarray:
    from supermimic.planner.features import quat_rotate_inverse

    q = L["root_quat"].astype(float)
    gz = quat_rotate_inverse(q, np.broadcast_to(np.array([0.0, 0.0, -1.0]), q.shape[:-1] + (3,)))[..., 2]
    fallen = (gz > -math.cos(fall_tilt)) | (L["root_pos"][..., 2] < fall_height)
    return np.array([int(np.argmax(fallen[:, e])) if fallen[:, e].any() else -1 for e in range(fallen.shape[1])])


def _gain(v, c, m):
    return float((v[m] * c[m]).sum() / max((c[m] ** 2).sum(), 1e-9)) if m.any() else float("nan")


def gain_table(v: np.ndarray, vg: np.ndarray, cmd: np.ndarray, m: np.ndarray) -> dict:
    """Gains (robot / command, goal / command) per axis and command kind on steady frames."""
    trans = np.abs(cmd[:, :2]).max(1) >= GAIN_MIN_CMD
    turn = np.abs(cmd[:, 2]) >= GAIN_MIN_CMD
    small_turn = np.abs(cmd[:, 2]) < 0.05
    small_trans = np.abs(cmd[:, :2]).max(1) < 0.05
    kinds = {"pure_turn": m & small_trans & turn, "pure_translation": m & small_turn & trans, "turn_while_walking": m & turn & trans, "all": m}
    out = {}
    for kind, mk in kinds.items():
        row = {"frames": int(mk.sum())}
        for i, ax in enumerate(("vx", "vy", "wz")):
            ma = mk & (np.abs(cmd[:, i]) >= GAIN_MIN_CMD)
            row[ax] = {"robot": _gain(v[:, i], cmd[:, i], ma), "goal": _gain(vg[:, i], cmd[:, i], ma), "frames": int(ma.sum())}
        out[kind] = row
    return out


def score(L: dict, fall_step: np.ndarray, settle_s: float, effort: np.ndarray) -> tuple[dict, list]:
    steps, n = L["joint"].shape[:2]
    settle = round(settle_s * 50)
    ve, V, VG, CMD, M, je, he, te, sat, peak, per_env = [], [], [], [], [], [], [], [], [], [], []
    for e in range(n):
        T = steps if fall_step[e] < 0 else fall_step[e]
        if T < 2 * settle:
            per_env.append({"env": e, "fell_at_s": fall_step[e] / 50, "frames": int(T)})
            continue
        rp, rq, q = L["root_pos"][:T, e].astype(float), L["root_quat"][:T, e].astype(float), L["joint"][:T, e]
        cmd = L["cmd"][:T, e].astype(float)
        g0 = L["goal0"][:T, e].astype(float)
        v = heading_velocity(rp, rq)
        # The goal's own velocity (heading-frame vx, vy and world yaw rate of its current frame),
        # through the same centered 25-frame mean as the robot's.
        k = np.ones(25)
        den = np.convolve(np.ones(T), k, mode="same")
        raw = np.concatenate([g0[:, G_LIN_H][:, :2], L["goal_wz"][:T, e, None]], 1)
        vg = np.stack([np.convolve(raw[:, j], k, mode="same") / den for j in range(3)], 1)
        m = steady_mask(cmd, settle)
        m[:13] = m[-13:] = False
        ve.append(np.abs(v[m] - cmd[m]))
        V.append(v)
        VG.append(vg)
        CMD.append(cmd)
        M.append(m)
        je.append(np.sqrt(((q - (g0[:, G_JOINT] + HOME_VEC)) ** 2).mean(1)))
        he.append(np.abs(rp[:, 2] - g0[:, G_HEIGHT]))
        grav_r = features_from_pose(rp, rq, q, HOME_VEC)[:, NJ + 1 : NJ + 4]
        gg = g0[:, G_GRAV]
        te.append(np.degrees(np.arccos(np.clip((grav_r * gg).sum(1) / np.linalg.norm(gg, axis=1), -1, 1))))
        frac = np.abs(L["tau"][:T, e]) / effort
        sat.append((frac > 0.99).any(1).mean())
        peak.append(float(frac.max()))
        err = ve[-1]
        per_env.append({"env": e, "fell_at_s": None if fall_step[e] < 0 else fall_step[e] / 50, "frames": int(T), "steady_vel_err": err.mean(0).round(3).tolist() if len(err) else None,
                        "saturated_pct": round(100 * float(sat[-1]), 2), "peak_tau_frac": round(peak[-1], 3)})
    cat = lambda xs: np.concatenate(xs) if xs else np.full((1, 3), np.nan)
    ve = cat(ve)
    je, he, te = (np.concatenate(x) if x else np.array([np.nan]) for x in (je, he, te))
    res = {
        "envs": n, "falls": int((fall_step >= 0).sum()), "fall_times_s": [round(s / 50, 2) for s in fall_step if s >= 0],
        "steady_vel_err_mean": {"vx_mps": float(ve[:, 0].mean()), "vy_mps": float(ve[:, 1].mean()), "wz_rps": float(ve[:, 2].mean())},
        "steady_vel_err_p90": {"vx_mps": float(np.percentile(ve[:, 0], 90)), "vy_mps": float(np.percentile(ve[:, 1], 90)), "wz_rps": float(np.percentile(ve[:, 2], 90))},
        "gains": gain_table(cat(V), cat(VG), cat(CMD), np.concatenate(M) if M else np.zeros(1, bool)),
        "tracking": {"joint_rmse_rad": float(je.mean()), "root_height_err_mm": float(he.mean() * 1000), "tilt_err_deg": float(te.mean())},
        "torque": {"saturated_step_pct": float(100 * np.mean(sat)) if sat else None, "peak_tau_over_limit": float(max(peak)) if peak else None},
    }
    return res, per_env


def goal_yaw_rate(L: dict) -> np.ndarray:
    """(steps, n) world yaw rate of the goal's current frame. The goal features carry the body
    angular velocity, not the world yaw rate; the world z axis in the body frame is -g_b, so the
    world z component of the angular velocity is ang_b . (-g_b) (exact; equal to the heading's rate
    when roll and pitch are steady)."""
    g0 = L["goal0"].astype(float)
    ang_b = g0[..., 2 * NJ + 3 : 2 * NJ + 6]
    up_b = -g0[..., G_GRAV] / np.linalg.norm(g0[..., G_GRAV], axis=-1, keepdims=True)
    return (ang_b * up_b).sum(-1)


def main(planner_run: str = "rover_v2_l1", planner_ckpt: str = "planner_last.pt", ckpt: Path = ROOT / "logs" / "rover_r2" / "ckpt_latest.pt", sources: tuple[str, ...] = ("plan", "data"),
         clips: int = 25, start: int = 60, seconds: float = 60.0, blend: int = 5, dr: bool = False, settle: float = 1.5, fall_tilt: float = math.radians(60), fall_height: float = 0.2,
         val_every: int = 10, device: str = "cuda:0", seed: int = 7, save: int = 6, out: Path | None = None) -> None:
    planner, ck_planner = load_planner(ROOT / "logs" / "planner" / planner_run / planner_ckpt, device)
    policy, pcfg = load_policy_from_checkpoint(ckpt, device)
    assert pcfg["robot"] == "rover", f"{ckpt} is a {pcfg['robot']} tracker"
    npz_dir = ROOT / ck_planner.get("npz_dir", "data/planner/rover/npz")
    files = [f for f in sorted(npz_dir.glob("*.npz")) if int(f.stem.split("_e")[1].split("_")[0]) % val_every == val_every - 1][:clips]
    if not files:  # no walking data: every robot starts from the run's standing seed, the planner's own output only
        files = [ROOT / "logs" / "planner" / planner_run / "stand_seed.npz"] * clips
        sources = tuple(s for s in sources if s != "data")
        print(f"no walking clips in {npz_dir}: {clips} robots from {files[0].name}, sources {sources}")
    out = out or ROOT / "logs" / "planner" / planner_run / f"closed_loop_{Path(ckpt).parent.name}{'_dr' if dr else ''}"
    out.mkdir(parents=True, exist_ok=True)
    spec = get_robot_spec("rover")
    fk = spec.load_model()
    effort = np.array([EFFORT_KNEE if "calf" in j else EFFORT_HIP for j in JOINT_NAMES])
    results = {"planner": f"{planner_run}/{planner_ckpt}", "label": ck_planner.get("label", "cmd"), "tracker": str(ckpt), "dr": dr, "start_frame": start, "seconds": seconds, "clips": [f.stem for f in files],
               "fall": {"tilt_deg": math.degrees(fall_tilt), "height_m": fall_height}}
    for source in sources:
        L = run(source, policy, planner, ck_planner, files, start, seconds, blend, dr, device, seed)
        L["goal_wz"] = goal_yaw_rate(L)
        fall_step = falls(L, fall_tilt, fall_height)
        res, per_env = score(L, fall_step, settle, effort)
        results[source] = {**res, "per_env": per_env}
        s = res
        g = s["gains"]
        print(f"[{source}] falls {s['falls']}/{s['envs']} {s['fall_times_s']} | steady |v-cmd| vx {s['steady_vel_err_mean']['vx_mps']:.3f} vy {s['steady_vel_err_mean']['vy_mps']:.3f} m/s "
              f"wz {s['steady_vel_err_mean']['wz_rps']:.3f} rad/s | joint rmse {s['tracking']['joint_rmse_rad']:.3f} rad, height err {s['tracking']['root_height_err_mm']:.1f} mm, "
              f"tilt err {s['tracking']['tilt_err_deg']:.1f} deg | torque saturated {s['torque']['saturated_step_pct']:.2f}% of steps, peak {s['torque']['peak_tau_over_limit']:.2f} of limit", flush=True)
        for kind, row in g.items():
            print(f"    gain {kind:20s} ({row['frames']:6d} fr) " + " ".join(f"{ax} robot {row[ax]['robot']:.2f} / goal {row[ax]['goal']:.2f} ({row[ax]['frames']})" for ax in ("vx", "vy", "wz")), flush=True)
        for e in range(min(save, len(files))):
            T = L["joint"].shape[0] if fall_step[e] < 0 else max(fall_step[e] + 50, 2)
            T = min(T, L["joint"].shape[0])
            clip = body_states_from_pose(fk, spec, L["root_pos"][:T, e].astype(float), L["root_quat"][:T, e].astype(float), L["joint"][:T, e].astype(float), 1 / 50)
            np.savez_compressed(out / f"{source}_robot_{e:02d}.npz", **clip.to_npz_dict(), cmd=L["cmd"][:T, e].astype(np.float32))
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    tyro.cli(main)
