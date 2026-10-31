# SuperMimic-Quadruped

[English](README.md) · **简体中文**

用 [SuperMimic](https://arxiv.org/abs/2610.28480)（一个学出来的可微世界模型 + 在想象轨迹上做短视野反传）给四足机器人做动作跟踪。
把公开的狗动作捕捉数据重定向到 12 自由度的四足机器人上，用**一个**策略跟踪全部动作。参考机器人是**越疆 Rover X1**；
重定向工具只需一个配置文件就能换成其他 12 自由度机器狗（附 Unitree Go1 示例）。

## 效果展示

银色：策略在 MuJoCo Warp 物理仿真中的动作；蓝色：它要跟踪的参考动作（重定向后的狗动捕），与机器狗画在同一位置，只在和机器狗不重合的地方露出来。评测口径：确定性策略，
无噪声、无随机化。"return" 是论文中的归一化跟踪回报（1.0 表示每一步都完美跟踪）。

| | | |
|:---:|:---:|:---:|
| <img src="assets/showcase/run.gif" width="290"/> | <img src="assets/showcase/zigzag.gif" width="290"/> | <img src="assets/showcase/turn_tilt.gif" width="290"/> |
| **奔跑**，最高 2.9 m/s<br><sub>腾讯拉布拉多 · return 0.67</sub> | **之字形急转**，偏航角速度最高 4 rad/s<br><sub>腾讯拉布拉多 · return 0.90</sub> | **大角度转身，机身倾斜 35°**<br><sub>AI4Animation · return 0.95</sub> |
| <img src="assets/showcase/spin.gif" width="290"/> | <img src="assets/showcase/rise.gif" width="290"/> | <img src="assets/showcase/holdout_turn.gif" width="290"/> |
| **原地转身**，2.4 秒转 140°<br><sub>AI4Animation · return 0.86</sub> | **从低姿态站起**，抬高 20 cm<br><sub>AI4Animation · return 0.97</sub> | **小跑转身（留出集，训练时从未见过）**<br><sub>AI4Animation · return 0.91</sub> |

## 结果

一次训练（5 万次迭代，单张 H200 上 4.6 小时，20,480 个并行环境），在动作库的每一段上评测（片段数含左右镜像）：

| 分组 | 片段 | 失败 | 平均 return | 含义 |
|---|---|---|---|---|
| train | 251 | 0 | 0.937 | 开环筛选认为容易的片段 |
| hard | 237 | 0 | 0.838 | 开环回放会摔倒的片段（多为小跑和奔跑），以一半权重参与训练 |
| holdout | 78 | 0 | 0.876 | 整条录制留出，不参与训练 |

"失败"指触发了训练时的终止条件（机身高度偏离参考 0.15 m，或倾角偏离 60°）。最弱的片段（return 0.45–0.5）是腾讯数据里的高跳、
走跑切换和一段快速的留出小跑，但都没有摔倒。

## 项目内容

```
公开动捕 ──► 重定向 (quadmotion) ──► 检查 ──► 筛选 ──► 分组 ──► 训练 (SuperMimic) ──► 评测 / 渲染
3 个来源      IK、脚步锁定、           FK、     开环 PD   train /   世界模型 +             benchmark、
约 470 MB     Froude 时间缩放、        脚、     分窗回放  hard /    在想象轨迹上反传        录像、GIF
              切段、镜像               限位               holdout
```

- **数据**（`scripts/data/`、`src/quadmotion/`）：三个公开来源，按固定版本下载并校验 SHA-256（`configs/sources.yaml`）：

  | 来源 | 原始数据 | 在 Rover X1 上保留 |
  |---|---|---|
  | AI4Animation 狗动捕（SIGGRAPH 2018） | 51 条 | 191 段，23.1 分钟 |
  | 腾讯拉布拉多动捕（NMI 2024） | 33 条 | 82 段，8.5 分钟 |
  | 小米 CyberDog2 动作表 | 37 个 | 10 段，1.0 分钟 |
  | **合计** | | **283 段，32.5 分钟（含镜像 566 段，65 分钟）** |

  狗动捕的处理流程：取肩、髋和四个爪子的关键点 → 建立机身坐标系 → 按站立高度缩放到机器人尺寸 → 时间按 √缩放比 拉伸，
  让腾空阶段仍符合重力 → 检测着地的爪子并锁定 → 解析 IK（膝盖朝后）→ 逐帧检查（可达性、关节限位和速度、机身和膝盖不触地、
  倾角、脚交叉、腾空、速度）→ 在不合格的帧处切段 → 左右镜像。CyberDog2 动作表逐步解释执行，并降低幅度直到每一帧都通过检查。
- **筛选**（`scripts/data/screen_clips.py`）：每 2 秒一个窗口做开环 PD 回放，超过 20% 的窗口摔倒的片段记为 `hard`。
  每个来源约 10% 的整条录制留作 holdout。
- **训练**（`src/supermimic/`）：基于 [mjlab](https://github.com/mujocolab/mjlab) / MuJoCo Warp 的 SuperMimic。
  世界模型用采集到的 8 步窗口学习仿真器的动力学；策略通过在世界模型里想象 32 步、把解析的跟踪奖励反传回来更新；
  critic 负责估计后续回报。策略只看机器人自己能测到的量（关节角和角速度、机身角速度、重力方向、上一步动作，10 帧历史）
  加上 10 帧未来参考动作。
- **机器人**（`src/supermimic/robots/rover.py`）：Rover X1 模型（网格和许可证来自越疆官方的
  [dobot_rover_simulation](https://github.com/Dobot-Team/dobot_rover_simulation)），PD 增益 kp 25 / kd 1.3（来自本地训练的 RL 行走策略），
  扭矩上限 23 / 23 / 55 N·m，随机化足端摩擦、编码器偏置和机身质心。spec 里每个数字都注明了出处。

## 快速开始

需要 Linux、NVIDIA GPU、Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync

# 数据：下载（约 470 MB）、重定向、检查、筛选（需 GPU）、分组
uv run python scripts/data/fetch_sources.py
uv run python scripts/data/build_dataset.py --robot rover
uv run python scripts/data/check_library.py --robot rover
uv run python scripts/data/screen_clips.py --robot rover
uv run python scripts/data/assign_groups.py --robot rover

# 训练（即上面结果的配置，约需 72 GiB 显存）
uv run sm-train --run-name rover --robot rover \
  --data.npz-dir data/npz/rover --data.manifest data/npz/rover/manifest.csv --data.group train,hard \
  --env.clip-weight-alpha 0.5 --env.static-bin-factor 0.25 \
  --env.num-envs 20480 --wm-trainer.batch-size 10240 --policy-trainer.batch-size 20480 \
  --schedule.total-iterations 50000 --schedule.lr-decay-fraction 1.0

# 评测、录制、渲染
uv run python scripts/eval_benchmark.py --ckpt logs/rover/ckpt_latest.pt \
  --npz-dir data/npz/rover --manifest data/npz/rover/manifest.csv --group holdout
uv run python scripts/record_rollouts.py --ckpt logs/rover/ckpt_latest.pt \
  --npz-dir data/npz/rover --manifest data/npz/rover/manifest.csv --exclude '_mirror$'
MUJOCO_GL=egl uv run python scripts/render_rollouts.py --rollouts logs/rover/rollouts_ckpt_latest.npz \
  tencent_dog_zig_walk_002_s16
```

显存较小时，把三个规模参数一起调小：`--env.num-envs 1024 --wm-trainer.batch-size 1024 --policy-trainer.batch-size 1024`
在 12 GB 的 RTX 5070 上约占 5.5 GiB、约 6 it/s（只做过短时间的冒烟测试；完整训练到最后的只有上面的完整配置）。`uv run pytest` 运行测试（需要动作库或 GPU 的测试在缺少时会跳过）。

## 换成其他机器人

见 [docs/PORTING.zh-CN.md](docs/PORTING.zh-CN.md)：每个机器人一个 YAML 文件用于重定向（`configs/robots/`），一个 Python spec
用于训练（`src/supermimic/robots/`）。重定向支持腿部结构为"绕 x 轴的外展关节 + 两个绕 y 轴的俯仰关节"的 12 自由度四足
（Rover X1、Go1、Go2、A1 等）。作为验证，`configs/robots/go1.yaml` 把同样的数据转换成了 mjlab 自带 Unitree Go1 的 520 段
（57.6 分钟），全部通过 `check_library.py`；Go1 暂时还没有训练 spec。

## 目录结构

```
configs/robots/        每个机器人的重定向描述（rover.yaml、go1.yaml）
configs/sources.yaml   固定版本的数据来源：每个文件的 URL、许可证、SHA-256
scripts/data/          fetch_sources、build_dataset、check_library、screen_clips、assign_groups、render_preview
scripts/               eval_benchmark、record_rollouts、render_rollouts、calibrate_scales、play
src/quadmotion/        重定向：机器人模型与 IK、BVH 读取、狗动捕、CyberDog2 解释器、片段读写
src/supermimic/        SuperMimic：环境、世界模型、想象 rollout、训练循环、机器人 spec（rover、g1）
docs/                  PORTING.md
assets/showcase/       上面的 GIF
```

## 致谢与许可

- SuperMimic：Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath, *SuperMimic: Scaling Humanoid Whole-Body
  Control with One Differentiable World Model*，[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)。
- 训练代码基于 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)（MIT，© Yuxi Liu）的 G1 复现，
  包括 [Mr-Zqr/SuperMimic-Reproduced](https://github.com/Mr-Zqr/SuperMimic-Reproduced) 的倒地起身扩展。世界模型、想象 rollout
  和训练循环沿用上游实现；本仓库新增了面向四足的机器人抽象、Rover X1 spec、重定向与数据管线，以及渲染工具。
  `src/supermimic/robots/g1.py` 保留为上游人形机器人的参考实现，它的测试需要上游的 LAFAN1 数据，缺少时自动跳过。
- 仿真：[mjlab](https://github.com/mujocolab/mjlab) 和 MuJoCo Warp。机器人模型：网格和许可证来自越疆官方的 [dobot_rover_simulation](https://github.com/Dobot-Team/dobot_rover_simulation)（BSD-3-Clause，© Dobot Robotics）。
- 代码采用 MIT 许可（`LICENSE`）。本仓库**不包含**动作数据；用这些数据生成的任何东西都受原始来源许可约束，
  其中 AI4Animation 为 CC BY-NC 4.0，不可商用。详见 `THIRD_PARTY_NOTICES.md`。
