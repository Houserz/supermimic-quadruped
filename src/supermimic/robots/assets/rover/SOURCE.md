# Dobot Rover X1 model — source

Official Dobot project: https://github.com/Dobot-Team/dobot_rover_simulation (commit a0a3173ee6fbb13edc6f3c2123409422e273e26a).

- `assets/*.STL` and `LICENSE` come from `dobot_rl_gym/resources/robots/dobot/meshes/` and `dobot_rl_gym/LICENSE` there.
- `dobot.xml` (SHA-256 ebb45f4cd4697cef2f24659675affbd788bdc25d49a4e2f115ab81e514b7fd55) is the MJCF every motion
  library and result in this repository uses. Its collision geoms are named and each foot carries a site (`FL`, `FR`,
  `RL`, `RR`). It has no actuators (passive joint damping 0.02 only); the PD actuators come from `rover.py`.
- Licence: BSD-3-Clause, Copyright (c) 2025 Dobot Robotics (`LICENSE`).
