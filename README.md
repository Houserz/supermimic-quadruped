# SuperMimic-Quadruped

**English** · [简体中文](README.zh-CN.md)

Motion tracking and joystick walking for quadruped robots with [SuperMimic](https://arxiv.org/abs/2610.28480)
(one learned differentiable world model + short-horizon backpropagation through imagined rollouts). Public dog
motion capture is retargeted to a 12-DoF quadruped, and one policy learns to track all of it. A SONIC-style
kinematic planner turns joystick input into reference motion in real time, and the same tracking policy
follows it. The reference robot is the **Dobot Rover X1**; the retargeting tools take other 12-DoF dogs from a
small config file (Unitree Go1 included as an example).

**News (2026-10-31):** joystick walking planner; tracker `rover_r2` trained on a 665-clip library; both
released as [pretrained models](#run-the-released-models).

## Joystick control

<img src="assets/showcase/joystick.gif" width="640"/>

The keys (bottom left, lit while held) set the stick; the planner turns the stick into the next second of
reference motion, replanning every 0.1 s; the tracking policy follows it in physics. **Blue:** the planner's
output, drawn beside the robot, with dots for its current plan. **Silver:** the tracking policy in MuJoCo Warp.
Arrows walk and turn, Shift + Left / Right walks sideways; the stick is the robot's velocity, up to 1.5 m/s
forward, 0.8 m/s back, 0.5 m/s sideways and 0.9 rad/s. Real time, from `scripts/planner/render_teleop.py`.

Five more keys play skills from the motion library (sit, shake hands, high five, stretch, a pose-medley
dance): the robot stops, blends into the clip, plays it, blends back to standing and walks on.

## Motion tracking

Silver: the policy, simulated in MuJoCo Warp. Blue: the reference motion it tracks, drawn at the same place
and visible only where it differs from the robot. Benchmark protocol: deterministic policy, no noise or
randomisation. "Return" is the paper's normalised tracking return (1.0 = perfect tracking at every step).

| | | |
|:---:|:---:|:---:|
| <img src="assets/showcase/zigzag.gif" width="290"/> | <img src="assets/showcase/spin.gif" width="290"/> | <img src="assets/showcase/run.gif" width="290"/> |
| **Zig-zag**, yaw rate up to 4 rad/s<br><sub>Tencent Labrador · return 0.89</sub> | **Turn in place**, 140° in 2.4 s<br><sub>AI4Animation · return 0.84</sub> | **Run**, up to 2.9 m/s<br><sub>Tencent Labrador · return 0.67</sub> |
| <img src="assets/showcase/lie.gif" width="290"/> | <img src="assets/showcase/rise.gif" width="290"/> | <img src="assets/showcase/holdout_turn.gif" width="290"/> |
| **Lie down and get up**, trunk down to 10 cm<br><sub>whole-body IK · return 0.99</sub> | **Get up from lying**, held-out take (never trained on)<br><sub>AI4Animation · return 0.98</sub> | **Trot and turn**, held-out take<br><sub>AI4Animation · return 0.91</sub> |

## Results

**Tracker** `rover_r2` (50k iterations, 4.6 h on one H200, 20,480 parallel environments), evaluated on every
clip of the library (counts include left-right mirrors):

| group | clips | failed | mean return | what it is |
|---|---|---|---|---|
| train | 307 | 0 | 0.937 | clips the open-loop screen finds easy |
| hard | 270 | 0 | 0.854 | clips whose open-loop replay falls (mostly trots and runs); trained at half weight |
| holdout | 88 | 0 | 0.891 | whole recordings left out of training |

A clip *fails* when a training termination fires (root height off by 0.15 m or tilt off by 60° from the
reference). The weakest clips (return 0.5-0.6) are fast Tencent runs, where the robot falls 10-20 cm behind
the reference; none of them falls.

**Joystick walking** (`rover_v2_l1` planner + `rover_r2`, `scripts/planner/closed_loop.py`): 125 robots,
60 s each, under a random joystick that ramps between held values anywhere in the stick's range:

| | clean | domain randomisation |
|---|---|---|
| falls | 0 | 0 |
| robot velocity / stick, vx / vy / yaw rate | 0.97 / 0.99 / 0.98 | 0.97 / 1.00 / 0.98 |
| steady-state error, vx / vy (m/s) / yaw rate (rad/s) | 0.018 / 0.010 / 0.015 | 0.020 / 0.014 / 0.022 |
| steps with a joint at its torque limit | 0 % | 0 % |

## What's inside

```
public mocap ──► retarget (quadmotion) ──► check ──► screen ──► groups ──► train (SuperMimic) ──► eval / render
3 sources        IK, foot locking,          FK,       open-loop   train /     world model +          benchmark,
~470 MB          Froude time, cuts,         feet,     PD replay   hard /      BPTT through           rollouts,
                 mirrors                    limits    windows     holdout     imagined rollouts      GIFs

keys ──► stick ──► walking planner ──► next 10 reference frames ──► tracking policy ──► robot
         ramps     MLP, 0.5 s context ──► 1 s plan, replanned every 0.1 s from its own output
```

- **Data** (`scripts/data/`, `src/quadmotion/`): three public sources, downloaded at pinned revisions with
  SHA-256 checks (`configs/sources.yaml`):

  | source | raw | kept on the Rover X1 |
  |---|---|---|
  | AI4Animation dog mocap (SIGGRAPH 2018) | 51 takes | 191 clips, 23.1 min |
  | Tencent Labrador mocap (NMI 2024) | 33 takes | 82 clips, 8.5 min |
  | Xiaomi CyberDog2 motion lists | 37 lists | 10 clips, 1.0 min |
  | **total** | | **283 clips, 32.5 min (566 / 65 min with mirrors)** |

  Dog clips: shoulder / hip and paw keypoints → trunk frame → scale to the robot's standing height → time
  stretched by √scale so that flight phases stay ballistic → planted paws detected and locked → analytic IK
  (knees backwards) → per-frame checks (reach, joint limits and speed, trunk and knees above the floor, tilt,
  crossed feet, flight, speed, stale source frames) → clips cut at failing frames → left-right mirrors.
  CyberDog2 motion lists are interpreted step by step and their amplitude lowered until every frame passes.

  `rover_r2` was also trained on 99 clips the pipeline here does not build, released as
  [`rover_r2_extra_clips.tar.gz`](#build-the-data-and-train): 78 walking clips of a locally trained RL
  walking policy (starts, stops, sideways, backwards, turns), 16 motions generated for the Rover with
  whole-body IK under a centre-of-mass support constraint (leg waves and lifts, moonwalk, lying down) and
  with optimal control in Crocoddyl (jumps, trot in place), 3 keyframed crouches and hops, and one
  motion_imitation reference motion (with its mirror). 665 clips, 87 min with mirrors.
- **Screen** (`scripts/data/screen_clips.py`): 2 s windows of open-loop PD replay; a clip whose windows fall
  more than 20 % of the time is `hard`. Whole recordings (~10 % of each source) are held out.
- **Training** (`src/supermimic/`): SuperMimic on [mjlab](https://github.com/mujocolab/mjlab) / MuJoCo Warp.
  A world model learns the simulator's dynamics from 8-step windows of collected transitions; the policy is
  improved by backpropagating the analytic tracking reward through 32-step rollouts imagined in that model;
  a critic bootstraps the return. The policy sees only what the robot measures (joint angles and velocities,
  angular velocity, gravity direction, last action; 10 frames of history) plus 10 future reference frames.
- **Walking planner** (`src/supermimic/planner/`, `scripts/planner/`, [docs/PLANNER.md](docs/PLANNER.md)): a
  residual MLP (10 M parameters) that takes the last 0.5 s of reference (joint angles, height, tilt, velocity
  in the heading frame, yaw rate) and the stick, and predicts the next 1 s. It runs autoregressively on its own
  output and never reads the robot, so the tracker sees a reference shaped like its training data. Trained on
  12.5 h of the RL walking policy under a random joystick (not included); the stick is in the robot's actual
  velocity (a fitted map of the policy's command to the velocity it reaches), and the walking policy's
  0.6 s gait clock is a second input, so gaits start from standing without sliding. A stand guard holds a
  still reference once the stick is released.
- **Robot** (`src/supermimic/robots/rover.py`): the Rover X1 model (meshes and licence from Dobot's official
  [dobot_rover_simulation](https://github.com/Dobot-Team/dobot_rover_simulation)), PD gains kp 25 / kd 1.3 (from a
  locally trained RL walking policy), torque limits 23 / 23 / 55 N·m, randomised foot friction, encoder bias and trunk COM. Every number
  in the spec says where it comes from.

## Quick start

Needs Linux, an NVIDIA GPU, Python 3.12 and [uv](https://docs.astral.sh/uv/).

### Run the released models

```bash
uv sync
# rover_r2 tracker (policy only), rover_v2_l1 planner and a standing start, into logs/ (64 MB)
curl -L https://github.com/Houserz/supermimic-quadruped/releases/download/v0.2.0/rover_models.tar.gz | tar xz

MUJOCO_GL=glfw uv run python scripts/planner/teleop.py                       # drive it from the keyboard
MUJOCO_GL=egl uv run python scripts/planner/render_teleop.py --out out/joystick.gif
uv run python scripts/planner/closed_loop.py --clips 25                      # random joystick, metrics
```

The skill keys need the CyberDog2 clips of the library (build it as below); walking works without it.

### Build the data and train

```bash
# Data: download (~470 MB), retarget, check, screen (GPU), assign groups
uv run python scripts/data/fetch_sources.py
uv run python scripts/data/build_dataset.py --robot rover
uv run python scripts/data/check_library.py --robot rover
uv run python scripts/data/screen_clips.py --robot rover
uv run python scripts/data/assign_groups.py --robot rover

# Optional: rover_r2's library = the 566 clips above + 99 extra clips and rover_r2's manifest (45 MB)
curl -L https://github.com/Houserz/supermimic-quadruped/releases/download/v0.2.0/rover_r2_extra_clips.tar.gz | tar xz
# ... then train with --data.manifest data/npz/rover/manifest_r2.csv

# Train (the configuration of the run above; ~72 GiB of GPU memory)
uv run sm-train --run-name rover --robot rover \
  --data.npz-dir data/npz/rover --data.manifest data/npz/rover/manifest.csv --data.group train,hard \
  --env.clip-weight-alpha 0.5 --env.static-bin-factor 0.25 \
  --env.num-envs 20480 --wm-trainer.batch-size 10240 --policy-trainer.batch-size 20480 \
  --schedule.total-iterations 50000 --schedule.lr-decay-fraction 1.0

# Evaluate, record and render
uv run python scripts/eval_benchmark.py --ckpt logs/rover/ckpt_latest.pt \
  --npz-dir data/npz/rover --manifest data/npz/rover/manifest.csv --group holdout
uv run python scripts/record_rollouts.py --ckpt logs/rover/ckpt_latest.pt \
  --npz-dir data/npz/rover --manifest data/npz/rover/manifest.csv --exclude '_mirror$'
MUJOCO_GL=egl uv run python scripts/render_rollouts.py --rollouts logs/rover/rollouts_ckpt_latest.npz \
  tencent_dog_zig_walk_002_s16
```

The released tracker is the policy only (no world model, critic or optimiser state), enough for every
script here but not for resuming training. On a smaller GPU, scale the three sizes down together:
`--env.num-envs 1024 --wm-trainer.batch-size 1024 --policy-trainer.batch-size 1024` runs in about 5.5 GiB at
~6 it/s on a 12 GB RTX 5070 (a short smoke run; only the full configuration above has been trained to the end).
`uv run pytest` runs the tests (those that need the library, the released models or a GPU skip without them).
Training a planner on your own walking data: [docs/PLANNER.md](docs/PLANNER.md).

## Other robots

[docs/PORTING.md](docs/PORTING.md): a YAML file per robot for the retargeting (`configs/robots/`), a Python
spec per robot for training (`src/supermimic/robots/`). The retargeting supports 12-DoF quadrupeds whose legs
are an abduction hinge about x and two pitch hinges about y (Rover X1, Go1, Go2, A1, ...). As a check,
`configs/robots/go1.yaml` turns the same sources into 520 clips (57.6 min) for mjlab's Unitree Go1, all passing
`check_library.py`; there is no Go1 training spec yet.

## Repository layout

```
configs/robots/          retargeting description per robot (rover.yaml, go1.yaml)
configs/sources.yaml     pinned sources: URL, licence, SHA-256 of every file
scripts/data/            fetch_sources, build_dataset, check_library, screen_clips, assign_groups, render_preview
scripts/planner/         teleop, render_teleop, closed_loop, train_planner, fit_velocity_map
scripts/                 eval_benchmark, record_rollouts, render_rollouts, calibrate_scales, play
src/quadmotion/          retargeting: robot model and IK, BVH reader, dog mocap, CyberDog2 interpreter, clip I/O
src/supermimic/          SuperMimic: env, world model, imagination, training loop, robot specs (rover, g1)
src/supermimic/planner/  walking planner, skills, keyboard and on-screen keys
docs/                    PORTING.md, PLANNER.md
assets/showcase/         the GIFs above
```

## Acknowledgements and licence

- SuperMimic: Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath, *SuperMimic: Scaling
  Humanoid Whole-Body Control with One Differentiable World Model*, [arXiv:2610.28480](https://arxiv.org/abs/2610.28480).
- The training code is built on [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
  (MIT, © Yuxi Liu), the G1 reproduction, including the get-up extension from
  [Mr-Zqr/SuperMimic-Reproduced](https://github.com/Mr-Zqr/SuperMimic-Reproduced). The world model, imagination
  and training loop are theirs; this repository adds the robot abstraction for the quadruped, the Rover X1 spec,
  the retargeting and data pipeline, the walking planner and skills, and the rendering.
  `src/supermimic/robots/g1.py` is kept as the upstream humanoid reference; its tests need the upstream LAFAN1
  data and skip without it.
- Simulation: [mjlab](https://github.com/mujocolab/mjlab) and MuJoCo Warp. Robot model: meshes and licence from Dobot's [dobot_rover_simulation](https://github.com/Dobot-Team/dobot_rover_simulation) (BSD-3-Clause, © Dobot Robotics).
  The generated extra clips were made with [Pinocchio](https://github.com/stack-of-tasks/pinocchio) and
  [Crocoddyl](https://github.com/loco-3d/crocoddyl).
- Code: MIT (`LICENSE`). Motion data is **not** in the repository; the sources' licences apply to anything
  built from them, and AI4Animation's (CC BY-NC 4.0) rules out commercial use. The released tracker was trained
  on that data, so the released models are for non-commercial research use. See `THIRD_PARTY_NOTICES.md`.
