# Walking planner and skills

**English** · [简体中文](PLANNER.zh-CN.md)

A kinematic planner in the style of SONIC: the joystick sets a velocity, the planner writes the reference
motion for the next second, and the motion-tracking policy follows it. The tracker is the same one that
tracks the mocap library; nothing is retrained for walking.

```
keys ─► stick target ─► RateLimiter ─► planner (every 5 steps) ─► reference queue ─► next 10 frames ─► tracker ─► robot
                         ≥ 0.2 s ramps   0.5 s context + stick      crossfade 5 frames    (goal features)
                                         + gait clock ─► 1 s plan
```

## Pieces

| file | what |
|---|---|
| `src/supermimic/planner/features.py` | the 20 numbers per frame: joint angles − home, root height, gravity in the root frame, velocity in the heading frame, yaw rate; and their integration back to a pose |
| `src/supermimic/planner/model.py` | residual MLP, 1024 × 4 blocks (10 M parameters): 25 context frames + condition → 50 frames, predicted as a residual on the last context frame |
| `src/supermimic/planner/goal.py` | plan frames → the tracker's goal features, through the motion library's own `goal_features` |
| `src/supermimic/planner/rover.py` | the Rover's home pose and feet, the random joystick, the gait clock, the velocity map, the stick's rate limiter |
| `src/supermimic/planner/rover_skills.py` | `SkillDriver`: walking from the planner, skills from library clips, every switch through standing; the stand guard |
| `src/supermimic/planner/session.py`, `keys.py`, `hud.py` | one robot driven by a stick, the keyboard, the on-screen keys |
| `scripts/planner/teleop.py` | live keyboard control in the MuJoCo viewer (`--selftest` runs scripted keys without a window) |
| `scripts/planner/render_teleop.py` | the demo video: scripted keys, planner beside the robot, keys on screen |
| `scripts/planner/closed_loop.py` | random joystick on many robots: falls, velocity gains, tracking error, torque |
| `scripts/planner/fit_velocity_map.py`, `train_planner.py` | training on your own walking data |

## Design

- **Local features, own output.** A frame holds no global position or heading, so a plan starts wherever the
  robot is. The planner's context is its own executed plan, never the robot's state: the tracker always gets
  a reference shaped like its training data, and a stumble does not feed back into the plan. Training makes
  the planner replan up to 20 times (2 s) from its own detached output, so it learns to stay on the data.
- **Gait clock.** The walking policy that produced the data has a free-running 0.6 s gait clock that is off
  below a command of 0.1. Which diagonal pair lifts first when a gait starts depends on that clock, so a
  planner that sees only the command has to guess, and averages the guesses into short, sliding steps
  (slow turns in place slid 18-21 cm/s against 3 in the data). The planner takes the clock (sin, cos) as two
  more inputs and runs its own copy at deployment.
- **The stick is velocity.** The walking policy reaches only ~0.88 / 0.71 / 0.92 of its command in vx / vy /
  yaw rate. `fit_velocity_map.py` fits the policy's steady-state map G (command → velocity, a polynomial)
  and its inverse; the planner is trained with G(command) as the label, so the stick value is the velocity
  the robot walks at. The clock is gated on G⁻¹(stick), the command that walks at that speed.
- **Stick ramps.** Every change of the stick target becomes a ramp of at least 0.2 s, all axes together,
  as in the recording. Faster steps are outside the data: a slow turn released in 0.1 s left the planner
  rocking in place.
- **Stand guard.** Once the stick is zero and the executed reference has stood for 0.5 s with every foot
  down, the planner is not called and the reference holds that frame until the stick moves. If the plan
  does not stand within 2 s it is blended to a standing frame (rare: a ~2.4 Hz yaw rocking after slow turns).
- **Skills.** A skill key ramps the stick to zero, waits until the plan stands, places the skill's library
  clip on the current feet (2-D rigid fit), blends into it with the feet planted (leg IK on a minimum-jerk root
  path), plays it, blends back to a standing frame and hands the reference back to the planner.

## Results (`rover_v2_l1` + `rover_r2`)

`closed_loop.py --val-every 1 --clips 125`: 125 robots for 60 s each, starting from validation clips of the
walking data, under the random joystick in the stick's full range (vx −0.8..1.5 m/s, vy ±0.5 m/s, yaw rate
±0.9 rad/s; holds of 1-5 s, ramps of 0.2-1 s):

| | clean | domain randomisation |
|---|---|---|
| falls | 0 | 0 |
| robot / stick, all frames vx / vy / yaw rate | 0.97 / 0.99 / 0.98 | 0.97 / 1.00 / 0.98 |
| robot / stick, pure sideways / pure turns | 0.99 / 0.98 | 1.00 / 0.97 |
| steady-state \|robot − stick\|, vx / vy (m/s) / yaw rate (rad/s) | 0.018 / 0.010 / 0.015 | 0.020 / 0.014 / 0.022 |
| tracking error to the plan: joints / height / tilt | 0.052 rad / 4.4 mm / 1.65° | 0.053 rad / 4.5 mm / 1.73° |
| steps with a joint at its torque limit | 0 % | 0 % |

The plan itself is at 1.00 of the stick; the 2-3 % left is the tracker. Without the walking data (the released
files), `closed_loop.py` starts every robot from `stand_seed.npz`: 32 robots × 30 s give 0 falls and
0.97 / 1.00 / 0.98. Known limits: turns at 0.25 rad/s come out ~9 % short; consecutive plans meet with a small
10 Hz joint-acceleration ripple; the yaw rocking above.

## Training on your own walking data

The walking data is not released (12.5 h of a locally trained RL walking policy). To train a planner for
another robot or policy, record your own:

1. Drive the walking policy with `planner.rover.RoverCommands` (each robot: hold a command for 1-5 s,
   ramp 0.2-1 s to the next; 15 % zero, 15 % pure turns, 25 % one translation axis; ramps capped at
   `ACC_CMD_V2`) and record at 50 Hz. Ours: 300 robots × 150 s, box vx −1..1.7, vy ±0.75, yaw rate ±1.
2. Save one npz per recorded segment as `<name>_e<episode>_s<k>.npz` in the mjlab motion layout (`joint_pos`,
   `joint_vel`, `body_pos_w`, `body_quat_w` (wxyz), `body_lin_vel_w`, `body_ang_vel_w`, `joint_names`,
   `body_names`, `fps`; the clip format of [PORTING.md](PORTING.md)) plus `cmd` (T, 3), the command given at
   each frame, and, for a clock-conditioned planner, `clock_step` (T,), the policy step of each frame. Episodes
   with `e % 10 == 9` are the validation split.
3. Fit the velocity map and train, then evaluate and drive it:

```bash
uv run python scripts/planner/fit_velocity_map.py --npz-dir data/planner/rover_v2/npz
uv run python scripts/planner/train_planner.py --npz-dir data/planner/rover_v2/npz --run my_planner   # ~26 min, RTX 5070
uv run python scripts/planner/closed_loop.py --planner-run my_planner --clips 125 --val-every 1
MUJOCO_GL=glfw uv run python scripts/planner/teleop.py --planner-run my_planner
```

Without a `stand_seed.npz` next to the checkpoint, the teleop starts from frame 60 of `rover_e009_s0` of the
walking data (`--clip` picks another) and takes the standing frame for the skills from the validation clips.
