# 移植到其他机器人

[English](PORTING.md) · **简体中文**

每个机器人两个文件：一个 YAML 告诉重定向腿在哪里，一个 Python spec 告诉训练这个机器人怎么驱动、怎么随机化、怎么打分。
Rover X1 两个都有（`configs/robots/rover.yaml`、`src/supermimic/robots/rover.py`）；Go1 只有第一个（`configs/robots/go1.yaml`），
已经足够生成它的动作库。

## 1. 重定向（狗动作 → 你的机器人）

支持的结构：12 自由度四足，MJCF 里有自由浮动的机身，每条腿依次是：在髋部 body 原点、绕 +x 的外展关节；相对它只有横向偏移、
绕 +y 的大腿关节；在大腿关节正下方、绕 +y 的小腿关节；以及小腿关节正下方的足端 site（Rover X1、Unitree Go1 / Go2 / A1
等大多数机器狗都是这样）。`load_robot` 会逐项检查，不符合时会指出是哪个关节或 site。

新建 `configs/robots/<name>.yaml`：

```yaml
xml: path/to/robot.xml            # 相对仓库根目录，或 mjlab:<mjlab 包内路径>
root_body: trunk
legs:                             # 腿: [外展关节, 大腿关节, 小腿关节, 足端 site]
  FL: [FL_hip_joint, FL_thigh_joint, FL_calf_joint, FL]
  FR: [FR_hip_joint, FR_thigh_joint, FR_calf_joint, FR]
  RL: [RL_hip_joint, RL_thigh_joint, RL_calf_joint, RL]
  RR: [RR_hip_joint, RR_thigh_joint, RR_calf_joint, RR]
geoms:
  foot: FL_foot_collision         # 球体，半径即足端半径
  trunk: trunk_collision          # 长方体，用来检查机身不触地
knee_sign: -1                     # -1：膝盖朝后（calf < 0）
stance: [0.9, -1.8]               # 正常站姿的 (大腿, 小腿) 角度，决定狗的动作缩放到多高
checks: {}                        # 可选：覆盖 quadmotion.dog.Checks 的阈值，例如 {max_joint_speed: 30.0}
```

腿长、髋部偏移、关节限位和足端半径都从 MJCF 读取。然后运行：

```bash
uv run pytest tests/test_quadmotion_robot.py       # 每个配置都会对照 MuJoCo 检查 FK / IK / 镜像，并重定向一段狗动作
uv run python scripts/data/build_dataset.py --robot <name>
uv run python scripts/data/check_library.py --robot <name>
MUJOCO_GL=egl uv run python scripts/data/render_preview.py --robot <name> --sampler 3
```

动作库输出到 `data/npz/<name>/`：npz 片段、`manifest.csv`、`clips.yaml`（记录每个被丢弃的片段及原因）、`report.md`。
片段里的关节按 MJCF 中的关节顺序存储。

## 2. 训练 spec（让 SuperMimic 认识你的机器人）

1. 把 `src/supermimic/robots/rover.py` 复制成 `<name>.py`，从头到尾改一遍：
   - 模型和名称：`xml_path`、`joint_names`（**必须是 MJCF 中的顺序**，也就是片段使用的顺序）、`root_body`、
     `tracked_bodies`（四足用机身加大腿、小腿即可）、`foot_geom_regex`；
   - 执行器：PD 增益、扭矩上限、动作缩放；
   - 终止条件：`root_height_threshold` / `root_tilt_threshold`；
   - 域随机化、观测和重置噪声、奖励的 std（Rover 的取值是把论文数值按站立高度缩放得到的，注释里写了怎么算）；
   - 镜像映射（`MIRROR_JOINT_PERM` / `MIRROR_JOINT_SIGN`、body 映射）和 `HOME` 站姿。
2. 在 `src/supermimic/robots/__init__.py` 里注册：`ROBOTS["<name>"] = "supermimic.robots.<name>:<name>_spec"`。
3. 测量世界模型的归一化尺度，填进 `state_scales`：
   `uv run python scripts/calibrate_scales.py --robot <name> --npz-dir data/npz/<name>`。
4. 筛选、分组，然后用 `--robot <name> --data.npz-dir data/npz/<name>` 训练（见 README 的"快速开始"）：
   `scripts/data/screen_clips.py --robot <name>`（spec 函数需要像 Rover 那样接受 `stiffness` 参数）和
   `scripts/data/assign_groups.py --robot <name>`。

## 3. 其他构型

训练部分并不假设机器人是四足：所有维度都由关节数推导，`src/supermimic/robots/g1.py`（上游的 Unitree G1 人形）就是第二个可用的 spec。
这里的重定向只负责把狗的动作映射到 12 自由度四足上。对于其他构型，按同样的 npz 格式自己准备片段（50 Hz；`joint_pos`、`joint_vel`、
`body_pos_w`、`body_quat_w`（wxyz）、`body_lin_vel_w`、`body_ang_vel_w`、`joint_names`、`body_names`、`fps`；body 按 MJCF 顺序、
不含 world），然后从第 2 步开始。
