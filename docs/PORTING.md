# Porting to another robot

**English** · [简体中文](PORTING.zh-CN.md)

Two files per robot: a YAML file that tells the retargeting where the legs are, and a Python spec that tells
the training how the robot is actuated, randomised and scored. The Rover X1 has both
(`configs/robots/rover.yaml`, `src/supermimic/robots/rover.py`); the Go1 has only the first
(`configs/robots/go1.yaml`), which is enough to build its motion library.

## 1. Retargeting (dog motions → your robot)

Supported: 12-DoF quadrupeds whose MJCF has a free-floating trunk and, per leg, an abduction hinge about +x at
the hip body origin, a thigh hinge about +y offset only sideways from it, a calf hinge about +y straight below
the thigh joint and a foot site straight below the calf joint (Rover X1, Unitree Go1 / Go2 / A1 and most
others). `load_robot` checks every one of these and names the joint or site that does not fit.

Write `configs/robots/<name>.yaml`:

```yaml
xml: path/to/robot.xml            # relative to the repo root, or mjlab:<path inside the mjlab package>
root_body: trunk
legs:                             # leg: [abad joint, thigh joint, calf joint, foot site]
  FL: [FL_hip_joint, FL_thigh_joint, FL_calf_joint, FL]
  FR: [FR_hip_joint, FR_thigh_joint, FR_calf_joint, FR]
  RL: [RL_hip_joint, RL_thigh_joint, RL_calf_joint, RL]
  RR: [RR_hip_joint, RR_thigh_joint, RR_calf_joint, RR]
geoms:
  foot: FL_foot_collision         # a sphere; its radius is the foot radius
  trunk: trunk_collision          # a box; used to keep the trunk off the floor
knee_sign: -1                     # -1: knees point backwards (calf < 0)
stance: [0.9, -1.8]               # (thigh, calf) of a normal standing pose; sets the height dogs are scaled to
checks: {}                        # optional overrides of quadmotion.dog.Checks, e.g. {max_joint_speed: 30.0}
```

Leg lengths, hip offsets, joint limits and the foot radius are read from the MJCF. Then:

```bash
uv run pytest tests/test_quadmotion_robot.py       # FK / IK / mirror against MuJoCo on every config, plus one dog clip
uv run python scripts/data/build_dataset.py --robot <name>
uv run python scripts/data/check_library.py --robot <name>
MUJOCO_GL=egl uv run python scripts/data/render_preview.py --robot <name> --sampler 3
```

The library lands in `data/npz/<name>/` (npz clips, `manifest.csv`, `clips.yaml` with every dropped segment
and why, `report.md`). Clips store joints in the MJCF's joint order.

## 2. Training spec (your robot in SuperMimic)

1. Copy `src/supermimic/robots/rover.py` to `<name>.py` and go through it top to bottom:
   - model and names: `xml_path`, `joint_names` (**in the MJCF's order**, the order the clips use),
     `root_body`, `tracked_bodies` (trunk plus thighs and calves works for a quadruped), `foot_geom_regex`;
   - actuators: PD gains, torque limits and action scale;
   - terminations: `root_height_threshold` / `root_tilt_threshold`;
   - domain randomisation, observation and reset noise, reward stds (the Rover's values scale the paper's
     by standing height; the comments say how);
   - mirror maps (`MIRROR_JOINT_PERM` / `MIRROR_JOINT_SIGN`, body map) and the `HOME` pose.
2. Register it in `src/supermimic/robots/__init__.py`: `ROBOTS["<name>"] = "supermimic.robots.<name>:<name>_spec"`.
3. Measure the world-model scales and copy them into `state_scales`:
   `uv run python scripts/calibrate_scales.py --robot <name> --npz-dir data/npz/<name>`.
4. Screen and group the library, then train with `--robot <name> --data.npz-dir data/npz/<name>`
   (README, *Quick start*):
   `scripts/data/screen_clips.py --robot <name>` (needs a `stiffness` argument in your spec function, as the
   Rover's has) and `scripts/data/assign_groups.py --robot <name>`.

## 3. Other body plans

The training side does not assume a quadruped: everything dimensional follows from the joint count, and
`src/supermimic/robots/g1.py` (the upstream Unitree G1 humanoid) is a second working spec. The retargeting
here only maps dog motions onto 12-DoF quadrupeds. For another body plan, write clips in the same npz layout
yourself (50 Hz; `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w` (wxyz), `body_lin_vel_w`,
`body_ang_vel_w`, `joint_names`, `body_names`, `fps`; bodies in MJCF order without the world body) and start
from step 2.
