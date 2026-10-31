# Third-party notices

## Code

| What | Where | Licence |
|---|---|---|
| SuperMimic reproduction (world model, imagination, training loop, tracking env) | `src/supermimic/`, derived from [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) and the get-up extension in [Mr-Zqr/SuperMimic-Reproduced](https://github.com/Mr-Zqr/SuperMimic-Reproduced) | MIT, Copyright (c) 2026 Yuxi Liu (`LICENSE`) |
| mjlab | dependency (pinned in `pyproject.toml`), not vendored | Apache-2.0 |

## Robot models

| What | Where | Licence |
|---|---|---|
| Dobot Rover X1 meshes (from [Dobot-Team/dobot_rover_simulation](https://github.com/Dobot-Team/dobot_rover_simulation)) and MJCF | `src/supermimic/robots/assets/rover/` (`SOURCE.md`) | BSD-3-Clause, Copyright (c) 2025 Dobot Robotics (`src/supermimic/robots/assets/rover/LICENSE`) |
| Unitree Go1 MJCF (retargeting example) | loaded from the installed mjlab package, not vendored | see mjlab |

## Motion data (not redistributed)

`scripts/data/fetch_sources.py` downloads these at pinned revisions; `configs/sources.yaml` records
the URL, licence and SHA-256 of every file. The repository itself holds no motion data and no trained
weights. Libraries you build from these sources, and policies trained on them, are derived from
them and carry their terms; in particular, AI4Animation's CC BY-NC 4.0 rules out commercial use.

| Source | Licence |
|---|---|
| AI4Animation dog motion capture: H. Zhang, S. Starke, T. Komura, J. Saito, *Mode-Adaptive Neural Networks for Quadruped Motion Control*, SIGGRAPH 2018 | CC BY-NC 4.0, research and education only |
| Tencent Robotics X Labrador motion capture: [lifelike-agility-and-play](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play) (Nature Machine Intelligence 2024) | MIT repository licence; its README restricts the code and data to research purposes |
| Xiaomi CyberDog2 motion lists: [cyberdog_locomotion](https://github.com/MiRoboticsLab/cyberdog_locomotion) | Apache-2.0 |

## Release files (GitHub release v0.2.0, not in the repository)

| File | What | Terms |
|---|---|---|
| `rover_models.tar.gz` | the `rover_r2` tracking policy, the `rover_v2_l1` walking planner, a 2 s standing start | trained on the library above (AI4Animation, Tencent) and on rollouts of a locally trained RL walking policy: non-commercial research use |
| `rover_r2_extra_clips.tar.gz` | 99 clips: RL walking rollouts, motions generated with whole-body IK ([Pinocchio](https://github.com/stack-of-tasks/pinocchio), BSD-2-Clause) and optimal control ([Crocoddyl](https://github.com/loco-3d/crocoddyl), BSD-3-Clause), keyframed motions, and `mi_inplace_steps` retargeted from a [motion_imitation](https://github.com/erwincoumans/motion_imitation) reference motion (X. B. Peng et al., *Learning Agile Robotic Locomotion Skills by Imitating Animals*, RSS 2020; Apache-2.0); the `qs_*` lifts and moonwalk follow the stepping of CyberDog2 motion lists (Apache-2.0) | MIT, except `mi_inplace_steps*` (Apache-2.0); details in its `EXTRA_CLIPS.md` |
