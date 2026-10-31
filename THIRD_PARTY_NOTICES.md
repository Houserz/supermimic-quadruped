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
the URL, licence and SHA-256 of every file. This repository ships no motion data and no trained
weights. Libraries you build from these sources, and policies trained on them, are derived from
them and carry their terms; in particular, AI4Animation's CC BY-NC 4.0 rules out commercial use.

| Source | Licence |
|---|---|
| AI4Animation dog motion capture: H. Zhang, S. Starke, T. Komura, J. Saito, *Mode-Adaptive Neural Networks for Quadruped Motion Control*, SIGGRAPH 2018 | CC BY-NC 4.0, research and education only |
| Tencent Robotics X Labrador motion capture: [lifelike-agility-and-play](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play) (Nature Machine Intelligence 2024) | MIT repository licence; its README restricts the code and data to research purposes |
| Xiaomi CyberDog2 motion lists: [cyberdog_locomotion](https://github.com/MiRoboticsLab/cyberdog_locomotion) | Apache-2.0 |
