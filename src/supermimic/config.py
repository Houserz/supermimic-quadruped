"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/rover")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep[,group,weight] (scripts/data/build_dataset.py output)
    group: str | None = None  # restrict to manifest rows whose group is in this comma list (e.g. "train" or "train,hard")
    exclude: str | None = None  # regex on clip names to leave out (e.g. "_mirror$" or "_e\\d+_s\\d+" for rollouts)
    library: Path | None = None  # packed compact library (.pt, upstream build_library.py); overrides npz_dir


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    replay_capacity: int = 1000
    future_frames: int = 10
    compact_library: bool = False
    anchor_obs: bool = False
    immune_probability: float = 0.0  # episodes without root terminations, for get-up training (plan D22)
    lie_probability: float = 0.0  # resets that start lying on the floor (D22)
    push_interval_s: tuple[float, float] | None = None  # knock-down pushes (D22); None = off
    push_speed: tuple[float, float] | None = None  # None = robot spec
    push_down_prob: float = 0.3
    clip_weight_alpha: float = 1.0  # plan 2.6 clip balancing: p(clip) ∝ duration^alpha
    static_bin_factor: float = 1.0  # plan 2.6: sampling weight of static (hold) bins
    support_foot_threshold: float | None = None  # None = robot spec; 0 = off
    # Root terminations (|ref - robot| root height / tilt). None = robot spec (rover: 0.15 m / 0.5).
    root_height_threshold: float | None = None
    root_tilt_threshold: float | None = None
    # Actuator overrides (None = robot spec; rover takes stiffness only). Stored in the checkpoint config and
    # re-applied by ``load_policy_from_checkpoint``.
    stiffness: float | None = None
    action_scale: float | None = None


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    # Iteration at which the cosine decay starts (lr = base before it). For annealing a resumed run: set it to the
    # checkpoint's iteration and total_iterations to the end of the anneal, e.g. 64000 / 104000 with decay_fraction 1.
    lr_decay_start: int = 0
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None


@dataclass
class TrainConfig:
    run_name: str = "debug"
    robot: str = "rover"  # robot spec name (supermimic.robots.ROBOTS)
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    tf32: bool = False  # TF32 matmuls (Ampere+): several x faster on H100/H200, slightly lower precision; off = fp32 numerics
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None
