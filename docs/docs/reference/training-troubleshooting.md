---
sidebar_position: 5
---

# Training Troubleshooting

Common training issues and solutions.

:::info
For training workflow, see [Training Tutorial](../tutorials/training). For data preparation, see [Dataset Reference](dataset).
:::

---

## Issue 1: Mean Episode Length = 1.00 (Robot Terminates on First Step)

### Symptoms

- `Mean episode length: 1.00`
- `Episode_Termination/anchor_pos` near total parallel environment count
- `Metrics/motion/error_anchor_pos` > 0.5 m
- `Metrics/motion/error_body_rot` very large (close to pi)

### Root Cause

Usually not a PPO hyperparameter issue, but **motion NPZ labels inconsistent with MuJoCo FK**:

1. **Body position coordinate error**: Local coordinates used as world coordinates
2. **Body order error**: Using PKL's 38-body order instead of mjlab G1's 30-body order
3. **Body orientation/angular velocity error**: All bodies approximated to root orientation

The current `convert_pkl_to_npz.py` fixes these issues.

### Quick Diagnosis

```bash
python train_mimic/scripts/data/check_motion_npz_fk.py \
    --npz data/lafan1_clips/lafan1/<clip>.npz
```

Expected thresholds: `pos_max < 1e-3 m`, `quat_mean < 0.05 rad`, `quat_p95 < 0.10 rad`.

If check fails, regenerate data and run a smoke test:

```bash
python train_mimic/scripts/train.py \
    --num_envs 64 --max_iterations 100 \
    --motion_file data/datasets/<dataset>
```

Expected: `Mean episode length` significantly > 1, `error_anchor_pos` starts decreasing.

---

## Issue 2: Episode Length Not Growing

### Symptoms

After 1000+ iterations, `Mean episode length` stays low (< 3) with no upward trend.

### Possible Causes

1. Poor retargeting quality (unreachable target poses)
2. Tracking reward weight too low vs regularization
3. Learning rate too high/low, clip_param mismatch
4. Termination thresholds too strict

### Diagnosis Steps

1. Visualize reference motion with `play.py`
2. Check reward distribution - tracking reward should dominate
3. Temporarily increase `bad_anchor_pos` threshold (0.25m -> 0.5m)
4. Compare with mjlab's built-in G1 tracking task

---

## Issue 3: Slow Training

### Symptoms

Training speed < 1000 steps/s (expected 1500-2000 on RTX 4090).

### Solutions

1. Increase `--num_envs` to 4096 (needs 24 GB VRAM)
2. Disable `--video` during training
3. Use TensorBoard instead of W&B (default)

---

## Issue 4: `nefc overflow - please increase njmax`

### Symptoms

```text
nefc overflow - please increase njmax to 257
```

### Root Cause

MuJoCo constraint buffer insufficient. When the robot falls or has many contacts, active constraints exceed `njmax`. The `mjlab` training default is `sim.njmax=250`.

### Solution

Already fixed in the repository. The env builder in `train_mimic/tasks/tracking/config/env.py` overrides:

```python
self.sim.njmax = 500
self.sim.nconmax = 150_000
```

If warnings persist at higher values, increase to `njmax = 800`.

:::note
Only modifying the robot XML is insufficient - the simulation-level `njmax` in mjlab takes precedence.
:::

---

## Issue 5: Benchmark Video Problems

### Video has only 1 frame

Ensure `num_eval_steps >= video_length`:

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/<dataset> \
    --num_envs 1 --num_eval_steps 2000 \
    --video --video_length 600
```

### EGL/OpenGL errors

Install OpenGL/EGL dependencies:

```bash
conda install -c conda-forge libopengl libglx libegl libglvnd pyopengl
```

If GPU EGL is unavailable, try CPU rendering:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    python train_mimic/scripts/benchmark.py ... --video
```

---

## Issue 6: Foot Sliding in Sim2Sim (Benchmark OK but ONNX Inference Slides)

### Root Cause

Sim2sim configuration parameters mismatch with training environment:

1. **`default_angles` mismatch (critical)**: Different joint defaults cause action offset and observation errors
2. **Missing joint armature**: Training environment has non-zero armature; zero armature causes overshoot
3. **condim mismatch**: Different collision parameters between training and sim2sim

### Diagnosis

```python
from mjlab.asset_zoo.robots import get_g1_robot_cfg
cfg = get_g1_robot_cfg()
print(cfg.init_state.joint_pos)  # Must match g1.yaml default_angles
```

### Solution

Update `teleopit/configs/robot/g1.yaml` and `assets/robots/unitree_g1/g1_29dof.xml` to match training environment values (default angles, armature, condim).

This fix also affects the sim2real path since `default_angles` is shared by `rl_policy.py` and `observation.py`.
