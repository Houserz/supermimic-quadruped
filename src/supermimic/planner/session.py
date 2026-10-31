"""One Rover driven by a joystick: walking planner + skills -> SuperMimic tracker, in one mjlab environment.

The loop of ``scripts/planner/teleop.py`` (live, keyboard) and ``scripts/planner/render_teleop.py``
(scripted keys, rendered to video): every control step the stick value goes through
``rover_skills.SkillDriver`` (rate limit, replanning every 5 steps from the planner's own plan,
skills), whose goal (the reference's next 10 frames) the tracker follows. Nothing in the loop reads
the robot: the planner sees only its own reference, the stick and its own gait clock.

The robot starts standing in a frame of a walking clip: ``stand_seed.npz`` next to the planner
checkpoint (shipped with the released models), else ``--clip`` of the planner's walking data.
Skills need their library clips in ``data/npz/rover`` (the CyberDog2 clips of the built library);
without them only walking is available.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from supermimic.planner.features import features_from_pose
from supermimic.planner.rover import HOME_VEC, joystick_cfg, load_planner
from supermimic.planner.rover_skills import (
    LIBRARY_DIR,
    M_SKILL,
    M_STOP,
    SEG_NAMES,
    SKILLS,
    SkillCfg,
    SkillDriver,
    SkillLibrary,
    hub_frame,
)

ROOT = Path(__file__).resolve().parents[3]


class TeleopSession:
    def __init__(self, planner_run: str = "rover_v2_l1", planner_ckpt: str = "planner_last.pt", ckpt: Path = ROOT / "logs" / "rover_r2" / "ckpt_latest.pt",
                 clip: str = "rover_e009_s0", start: int = 60, blend: int = 5, dr: bool = False, stand_hold: bool = True, accel=(2.0, 1.5, 2.0),
                 library_dir: Path = LIBRARY_DIR, device: str = "cuda:0"):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.utils.torch import configure_torch_backends

        from supermimic.envs import SuperMimicEnvCfgArgs
        from supermimic.envs.tracking_env_cfg import make_env_cfg
        from supermimic.eval.benchmark import load_policy_from_checkpoint

        run_dir = ROOT / "logs" / "planner" / planner_run
        self.planner, self.ck = load_planner(run_dir / planner_ckpt, device)
        self.policy, pcfg = load_policy_from_checkpoint(ckpt, device)
        assert pcfg["robot"] == "rover", f"{ckpt} is a {pcfg['robot']} tracker"
        data_dir = ROOT / self.ck.get("npz_dir", "data/planner/rover/npz")
        npz = run_dir / "stand_seed.npz"
        if not npz.exists():
            npz = data_dir / f"{clip}.npz"
        jc = joystick_cfg(self.ck)
        self.box = (jc.lo, jc.hi)  # the stick's box in the planner's units (actual velocity for rover_v2_l1)
        args = SuperMimicEnvCfgArgs(npz_files=[str(npz)], robot="rover", num_envs=1, future_frames=10, episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=dr,
                                    reset_noise=False, domain_randomization=dr, sampling_mode="start", record_transitions=False, seed=0, root_height_threshold=1.0e3,
                                    root_tilt_threshold=1.0e3, support_foot_threshold=0.0)
        cfg = make_env_cfg(args)
        cfg.terminations.pop("clip_end")  # the clip only seeds the reset
        configure_torch_backends()
        self.env = ManagerBasedRlEnv(cfg, device=device)
        self.cmd_term = self.env.command_manager.get_term("motion")
        self.lib = self.cmd_term.library
        self.robot = self.env.scene["robot"]
        self.start = start
        with np.load(npz) as z:
            self.clock0 = int(z["clock_step"][start]) if "clock_step" in z.files else 0  # the planner's gait clock runs on from here
            self.ctx0 = features_from_pose(z["body_pos_w"][:, 0].astype(float), z["body_quat_w"][:, 0].astype(float), z["joint_pos"].astype(float),
                                           HOME_VEC)[start - self.planner.cfg.context : start]
        hub = np.asarray(self.ck["hub"], np.float64) if "hub" in self.ck else hub_frame(data_dir)  # a standing frame of the planner's data
        available = {k: v for k, v in SKILLS.items() if all((Path(library_dir) / f"{c}.npz").exists() for c in v.clips)}
        if len(available) < len(SKILLS):
            print(f"[teleop] skills without their clips in {library_dir} are off: {sorted(set(SKILLS) - set(available))}")
        skills = SkillLibrary(available, SkillCfg(accel=tuple(accel), stand_hold=stand_hold), library_dir=library_dir, hub=hub)
        self.drv = SkillDriver(self.planner, self.ck, skills, 1, self.lib.default_joint_pos, device, blend=blend)
        self.reset()

    def reset(self) -> None:
        self.cmd_term.forced_frame[:] = self.lib.clip_start[0] + self.start
        self.obs, _ = self.env.reset()
        self.drv.reset(self.ctx0[None], np.array([self.clock0]))
        self.cmd = np.zeros(3)

    @torch.no_grad()
    def step(self, target: np.ndarray) -> None:
        """One control step (50 Hz) towards the stick value ``target`` (3,)."""
        goal = self.drv.step(np.asarray(target, np.float64)[None])
        self.obs, _, _, _, _ = self.env.step(self.policy(self.obs["policy"], goal))
        self.cmd = self.drv.cmd[0].copy()

    def request(self, skill: str) -> bool:
        return skill in self.drv.lib.names and self.drv.request(0, skill)

    # ---------------------------------------------------------------- read-outs
    def robot_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World root position (3), root quaternion wxyz (4), joint angles (12) of the simulated robot."""
        d = self.robot.data
        pos = (d.root_link_pos_w[0] - self.env.scene.env_origins[0]).cpu().numpy()
        return pos, d.root_link_quat_w[0].cpu().numpy(), d.joint_pos[0].cpu().numpy()

    def robot_velocity(self) -> np.ndarray:
        """(vx, vy) in the body frame and the yaw rate, like the stick."""
        d = self.robot.data
        return np.r_[d.root_link_lin_vel_b[0, :2].cpu().numpy(), float(d.root_link_ang_vel_b[0, 2])]

    def tilt_deg(self) -> float:
        q = self.robot.data.root_link_quat_w[0].cpu().numpy()
        return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))

    def reference(self) -> np.ndarray:
        """(20,) the reference frame executed this step (planner features)."""
        return self.drv.hist[0, -1].cpu().numpy().astype(np.float64)

    def upcoming(self, frames: int = 50) -> np.ndarray:
        """(frames, 20) the reference queued after it: the current plan."""
        return self.drv.queue[0, :frames].cpu().numpy().astype(np.float64)

    def status(self) -> str:
        drv = self.drv
        if drv.mode[0] == M_STOP:
            return f"stopping for {drv.lib.names[drv.skill[0]]}"
        if drv.mode[0] == M_SKILL:
            return f"{drv.lib.names[drv.skill[0]]}: {SEG_NAMES[drv.seg_now[0]]} ({drv.remaining[0] / 50:.1f} s left)"
        return "standing (held: no replanning until the stick moves)" if drv.held[0] else "walking"

    def close(self) -> None:
        self.env.close()
