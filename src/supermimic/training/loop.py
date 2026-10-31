"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz, manifest_clip_weights
from supermimic.envs.tracking_env_cfg import make_heads
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.policy.nets import Critic, Policy
from supermimic.robots import get_robot_spec, set_robot_overrides
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed, world_size
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


def apply_robot_defaults(cfg: TrainConfig) -> None:
    """Fill the robot-dependent parts of ``cfg`` from the robot spec (dims, scales, thresholds).

    Explicit CLI values survive where they differ from the dataclass defaults only for the
    world-model hidden sizes etc.; the dimensional fields are always derived.
    """
    set_robot_overrides(stiffness=cfg.env.stiffness, action_scale=cfg.env.action_scale)
    spec = get_robot_spec(cfg.robot)
    cfg.policy.nj = spec.nj
    cfg.policy.obs_term_scale = tuple(spec.obs_term_scale)
    cfg.policy.goal_term_scale = tuple(spec.goal_term_scale)
    cfg.policy.goal_dim = cfg.env.future_frames * spec.goal_feat_dim + (9 if cfg.env.anchor_obs else 0)
    cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
    cfg.policy_trainer.future_frames = cfg.env.future_frames
    cfg.world_model.state_spec = StateSpecCfg.from_robot(spec, dt=spec.sim.control_dt, explicit_integration=cfg.world_model.state_spec.explicit_integration)
    # The world model's termination head must use the same thresholds as the simulator.
    cfg.world_model.height_threshold = spec.root_height_threshold if cfg.env.root_height_threshold is None else cfg.env.root_height_threshold
    cfg.world_model.tilt_threshold = spec.root_tilt_threshold if cfg.env.root_tilt_threshold is None else cfg.env.root_tilt_threshold
    foot_thr = spec.support_foot_threshold if cfg.env.support_foot_threshold is None else cfg.env.support_foot_threshold
    cfg.world_model.support_foot_threshold = foot_thr if spec.support_feet else 0.0
    if not spec.has_action_delay or not cfg.env.domain_randomization:
        # xi carries the action delay only when the actuators randomise it.
        s = cfg.world_model.state_spec
        n = 1 + spec.nj + 3
        cfg.world_model.state_spec = StateSpecCfg(**{**asdict(s), "xi_offset": s.xi_offset[:n], "xi_scale": s.xi_scale[:n]})


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        apply_robot_defaults(cfg)
        spec = get_robot_spec(cfg.robot)
        self.spec = spec
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest, group=cfg.data.group, exclude=cfg.data.exclude)
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                robot=cfg.robot,
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                immune_probability=cfg.env.immune_probability,
                lie_probability=cfg.env.lie_probability,
                push_interval_s=cfg.env.push_interval_s,
                push_speed=cfg.env.push_speed,
                push_down_prob=cfg.env.push_down_prob,
                clip_weight_alpha=cfg.env.clip_weight_alpha,
                static_bin_factor=cfg.env.static_bin_factor,
                clip_weights=manifest_clip_weights(cfg.data.manifest),
                support_foot_threshold=cfg.env.support_foot_threshold,
                root_height_threshold=cfg.env.root_height_threshold,
                root_tilt_threshold=cfg.env.root_tilt_threshold,
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        assert self.recorder.xi_dim == cfg.world_model.state_spec.xi_dim, (self.recorder.xi_dim, cfg.world_model.state_spec.xi_dim)

        heads = make_heads(env, cfg.robot, self.device)
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, feat_dim=spec.feat_dim, xi_dim=cfg.world_model.state_spec.xi_dim, pcfg=cfg.policy).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.wm_trainer.dump_path = str(self.log_dir / "wm_nan_batch.pt")
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:
            print(f"[INFO] robot {cfg.robot}: nj {spec.nj}, state {spec.state_dim}, obs {spec.obs_dim}x{cfg.policy.history}, goal {cfg.policy.goal_dim}, xi {cfg.world_model.state_spec.xi_dim}")
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0
        self.num_updates = 0
        self._utd_acc = 0.0
        if cfg.resume is not None:
            self.load(cfg.resume)

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                "bin_return": self.command.bin_return,
                "bin_memory": self.command.bin_memory,
                "config": asdict(self.cfg),
            },
            self.log_dir / name,
        )

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        critic_upgraded = Critic.upgrade_state_dict(ck["critic"], feat_dim=self.critic.feat_dim, xi_dim=self.critic.xi_dim)
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        if critic_upgraded:
            # Adam moments of the old first layer no longer match its shape: restart the critic optimizer.
            print("[INFO] pre-D22 checkpoint: critic gets a zero-initialised immune input, critic optimizer reset")
        else:
            self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        if ck["bin_return"].shape == self.command.bin_return.shape:
            self.command.bin_return.copy_(ck["bin_return"])
            self.command.bin_memory.copy_(ck["bin_memory"])
        else:
            print("[INFO] checkpoint sampler state is for a different clip set; sampler restarted")
        self.iteration = ck["iteration"]
        self.num_updates = ck["num_updates"]
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        obs, _ = env.reset()
        # The replay buffer is not checkpointed: a resumed run refills it before its first update.
        start_iteration = self.iteration
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        immune_returns: list[float] = []  # immune episodes (D22) are logged apart from the paper metric
        immune_lens: list[float] = []
        last_log = time.time()

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            action = self.act(obs)
            immune = self.command.immune.clone()  # flags of the episodes this step belongs to (reset resamples them)
            obs, rew, term, tout, extras = env.step(action)
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if done.any():
                ids = done.nonzero().squeeze(-1)
                lens = self.ep_len[ids]
                norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                im = immune[ids]
                finished_returns.extend(norm_ret[~im].tolist())
                finished_lens.extend(lens[~im].tolist())
                immune_returns.extend(norm_ret[im].tolist())
                immune_lens.extend(lens[im].tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if self.iteration - start_iteration >= sch.warmup_iterations:
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr = cosine_lr(
                        self.iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction, sch.lr_decay_start
                    )
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                    t0 = time.perf_counter()
                    try:
                        m_wm = self.wm_trainer.update(self.buffer)
                    except RuntimeError as e:  # replay buffer temporarily without sampleable windows
                        if "no sampleable rows" not in str(e):
                            raise
                        print(f"[WARN] it {self.iteration}: {e}; update skipped")
                        continue
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    for _ in range(sch.policy_updates_per_wm_update):
                        m_pi = self.policy_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["lr"] = lr
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.time() - t_start
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = int(round(cnt))
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    immune_returns, immune_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                else:
                    tr = float("nan")
                if self.is_main and immune_returns:
                    self.writer.add_scalar("train/immune_episode_norm_return", sum(immune_returns) / len(immune_returns), it)
                    self.writer.add_scalar("train/immune_episode_len", sum(immune_lens) / len(immune_lens), it)
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    peak_gib = torch.cuda.max_memory_reserved(self.device) / 2**30  # sizing runs for a bigger GPU
                    self.writer.add_scalar("time/peak_mem_gib", peak_gib, it)
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    self.writer.add_scalar("train/immune_frac", float(self.command.immune.float().mean()), it)
                    self.writer.add_scalar("train/lying_frac", float(self.command.lying.float().mean()), it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s | mem {peak_gib:.1f} GiB"
                    )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                immune_returns, immune_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(stop)  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.writer is not None:
            self.writer.flush()
        finalize()
