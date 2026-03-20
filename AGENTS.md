# AGENTS.md

## Project Overview

Teleopit is a lightweight, extensible, self-contained humanoid robot whole-body teleoperation framework. It integrates GMR (General Motion Retargeting) and supports train_mimic-exported ONNX RL policy inference.

Language: Python 3.10+
Package: `teleopit` (installed via `pip install -e .`)
Config: Hydra/OmegaConf YAML files in `teleopit/configs/`

## Architecture

```
InputProvider (BVH file / UDP realtime / Pico4 VR) → Retargeter (GMR) → ObservationBuilder (166D VelCmdHistory or 1587D MotionTrackingDeploy) → Controller (dual-input TemporalCNN ONNX or single-input MLP ONNX) → Robot (MuJoCo + PD / Unitree SDK)
```

Module-internal isolation: all modules run in-process and communicate via `InProcessBus` (zero-copy). Core interfaces are defined as `typing.Protocol` in `teleopit/interfaces.py`.

## Supported Surface

- Default training task: `Tracking-Flat-G1-VelCmdHistory`
- Optional deploy-aligned training task: `Tracking-Flat-G1-MotionTrackingDeploy`
- Default inference observation: `velcmd_history` (166D, dual-input ONNX with `obs` + `obs_history`)
- Optional deploy-aligned inference observation: `motion_tracking_deploy` (1587D, single-input ONNX with `obs`)
- Safe runtime defaults stay on VelCmdHistory; deploy-aligned motion tracking is opt-in via dedicated Hydra configs
- Realtime inference uses a retargeted-reference timeline before observation build; `reference_steps=[0]` remains the default production path, while motion-tracking deploy configs opt into the deployed future/history window
- Adaptive sampling, teacher-student training, and old legacy task variants are removed

## Directory Structure

```
teleopit/                 # Core inference package
├── interfaces.py         # Protocol definitions: Robot, Controller, InputProvider, Retargeter, etc.
├── pipeline.py           # TeleopPipeline — thin sim runtime facade
├── runtime/              # Shared runtime assembly: config/path resolution, factories, CLI helpers
├── bus/                  # InProcessBus message pub/sub
├── configs/              # Hydra YAML configs
│   ├── default.yaml      # Offline VelCmdHistory sim2sim (safe default)
│   ├── online.yaml       # Online VelCmdHistory sim2sim (safe default)
│   ├── sim2real.yaml     # VelCmdHistory sim2real (safe default)
│   ├── motion_tracking*.yaml # Deploy-aligned motion-tracking entrypoints
│   ├── robot/g1.yaml     # G1 robot: XML path, PD gains, default angles, action dims
│   ├── controller/rl_policy.yaml
│   ├── controller/motion_tracking_policy.yaml
│   ├── input/bvh.yaml    # Offline BVH file input
│   └── input/udp_bvh.yaml # UDP realtime BVH input (reference_bvh, port, timeout)
├── controllers/
│   ├── rl_policy.py      # RLPolicyController — single-input or dual-input ONNX inference with fail-fast dim checks
│   └── observation.py    # VelCmdObservationBuilder + MotionTrackingObservationBuilder
├── inputs/
│   ├── bvh_provider.py       # BVHInputProvider — offline BVH file
│   ├── pico4_provider.py     # Pico4InputProvider — xrobotoolkit_sdk realtime body tracking input
│   ├── rot_utils.py          # Quaternion helpers for input-space transforms
│   └── udp_bvh_provider.py   # UDPBVHInputProvider — realtime UDP BVH receiver
├── retargeting/
│   ├── core.py           # RetargetingModule + extract_mimic_obs()
│   └── gmr/              # Self-contained GMR (assets, IK solver, robot configs)
├── robots/
│   └── mujoco_robot.py   # MuJoCoRobot — MuJoCo sim wrapper
├── sim/
│   └── loop.py           # SimulationLoop — PD control at 1000Hz, policy at 50Hz
└── recording/            # HDF5Recorder
scripts/
├── run_sim.py            # Offline / online sim2sim pipeline
├── run_sim2real.py       # G1 sim2real control; supports Pico4 via dedicated config names
├── send_bvh_udp.py       # UDP BVH test sender
├── render_sim.py         # Render single BVH → 3 videos (bvh skeleton, retarget, sim2sim)
├── compute_ik_offsets.py # Compute IK quaternion offsets for new BVH formats
└── setup_pico4.sh        # Pico4 environment setup helper
train_mimic/              # Training package
├── app.py                # Shared app helpers for train/play/benchmark
├── tasks/tracking/config/
│   ├── constants.py      # Public task constants
│   ├── registry.py       # Registers VelCmdHistory and MotionTrackingDeploy tasks
│   ├── env.py            # VelCmdHistory / MotionTrackingDeploy env builders
│   └── rl.py             # TemporalCNN PPO cfg + deploy MLP PPO cfg
├── tasks/tracking/rl/
│   ├── runner.py         # ONNX export wrapper for policy + motion labels
│   ├── conv1d_encoder.py # 1-D CNN encoder for temporal history groups
│   └── temporal_cnn_model.py # TemporalCNN actor/critic model
└── scripts/
    ├── train.py          # Training entry point
    ├── play.py           # Checkpoint playback
    ├── benchmark.py      # Policy evaluation with tracking errors
    └── save_onnx.py      # Export TemporalCNN or deploy MLP ONNX
```

## Key Technical Details

### Sim2Sim Pipeline
- Policy runs at 50Hz, PD control at 1000Hz (`decimation=20`, `sim_dt=0.001`)
- Action flow: `compute_action()` returns raw action → `get_target_dof_pos()` applies clip `[-10, 10]`, scale, and `default_dof_pos`
- Must use `g1_mjlab.xml` for sim2sim; `g1_mocap_29dof.xml` clamps torques to `±1 Nm` and is only for kinematic retarget visualization

### Multi-Viewer Support
`SimulationLoop` supports three simultaneous viewer windows controlled by the `viewers` config:

```bash
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=sim2sim
python scripts/run_sim.py controller.policy_path=policy.onnx 'viewers=[bvh,retarget,sim2sim]'
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=all
python scripts/run_sim.py controller.policy_path=policy.onnx 'viewers=[retarget,sim2sim]'
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=none
```

- `sim2sim`: MuJoCo physics result
- `retarget`: kinematic retarget result
- `bvh`: source BVH skeleton
- All viewers run in separate subprocesses because GLFW/GLX only supports one window per process
- Simulation exits when all active viewer windows are closed
- `viewers` is the only supported viewer key; legacy `viewer` alias is removed

### default_dof_pos Propagation
RL policy outputs action offsets relative to the default standing pose:

```
target_dof_pos = clip(action, -10, 10) × action_scale + default_dof_pos
```

`default_dof_pos` comes from `robot/g1.yaml` `default_angles`. `TeleopPipeline` automatically propagates `robot_cfg.default_angles` into `controller_cfg.default_dof_pos`. If this propagation is missing, knees and elbows lose their standing offset and the robot cannot balance.

### Online Sim2Sim (UDP realtime input)
- Realtime BVH arrives as one BVH motion line per UDP packet
- `UDPBVHInputProvider` parses skeleton metadata from `reference_bvh`
- A daemon thread receives packets and stores only the latest processed frame
- `get_frame()` always returns the newest frame; `is_available()` stays `True`
- `fps=30` is fixed for the UDP provider
- Realtime control writes retargeted `qpos` into a short reference timeline and samples from that timeline at `time.monotonic() - retarget_buffer_delay_s`
- `reference_steps=[0]` is the safe default path; deploy-aligned motion tracking uses `[0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16]`
- Non-zero `reference_steps` must satisfy `retarget_buffer_delay_s >= max_future_step / policy_hz` and `retarget_buffer_window_s >= retarget_buffer_delay_s + abs(min_history_step) / policy_hz`; runtime fails fast on misconfiguration instead of silently living on fallback samples

### Pico4 Realtime Input
- `Pico4InputProvider` reads realtime body tracking from `xrobotoolkit_sdk`
- Bone naming follows `xrobot_to_g1.json`
- The provider applies an input-space transform to match the current retarget config
- Do not hardcode that transform as a public coordinate-system contract; validate against actual retarget/sim2sim behavior when SDK or firmware changes
- Pico4 realtime control uses the same retargeted-reference timeline path as UDP realtime input

### SimulationLoop Runtime Behavior
- `realtime=true` enforces wall-clock pacing even without a viewer
- `num_steps=0` means infinite loop (`max_steps = 2**63`)
- `KeyboardInterrupt` is handled for clean shutdown
- BVH frame alignment is time-based: `bvh_idx = int(policy_time × input_fps)`
- Realtime reference buffering is controlled by `retarget_buffer_enabled`, `retarget_buffer_window_s`, `retarget_buffer_delay_s`, `reference_steps`, `realtime_buffer_warmup_steps`, and the low/high watermark knobs
- When the future horizon briefly runs dry, runtime only repeat-pads positive future steps in the sampled `ReferenceWindow`; it does not append synthetic frames into the timeline
- Realtime inferred `motion_joint_vel`, anchor linear velocity, and anchor angular velocity can be EMA-smoothed via `reference_velocity_smoothing_alpha` and `reference_anchor_velocity_smoothing_alpha`

### Inference Observation
Two inference observation families are supported:

1. `velcmd_history` (166D, dual-input ONNX)

```
command(58)
+ motion_anchor_ori_b(6)
+ base_ang_vel(3)
+ joint_pos_rel(29)
+ joint_vel(29)
+ last_action(29)
+ projected_gravity(3)
+ ref_base_lin_vel_b(3)
+ ref_base_ang_vel_b(3)
+ ref_projected_gravity_b(3)
```

2. `motion_tracking_deploy` (1587D, single-input ONNX)

```
boot_indicator(1)
+ tracking_command(96)
+ target_joint_pos(638)
+ target_root_z(11)
+ target_projected_gravity_b(33)
+ root_ang_vel_history(27)
+ projected_gravity_history(27)
+ joint_pos_history(261)
+ joint_vel_history(261)
+ prev_actions(232)
```

Deploy-aligned motion tracking follows the sibling `motion_tracking/sim2real` deployed policy semantics:
- `future_steps=[0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16]`
- `prev_action_steps=8`
- `root_angvel_history_steps = projected_gravity_history_steps = joint_pos_history_steps = joint_vel_history_steps = [0, 1, 2, 3, 4, 8, 12, 16, 20]`
- Teleopit intentionally removes `compliance_flag`; deploy observations no longer reserve that 3D constant term
- The runtime builder requires a real `reference_window` and raises on missing/mismatched windows instead of silently fabricating one
- The deploy builder intentionally does not depend on `base_pos` / `base_lin_vel`, so real-robot inference does not degenerate when those fields are unavailable

Runtime constraints:
- Public builders are `VelCmdObservationBuilder` and `MotionTrackingObservationBuilder`
- `RLPolicyController` accepts either single-input `obs` ONNX or dual-input `obs` + `obs_history` ONNX
- Startup validates the observation definition against the ONNX signature and raises immediately on mismatch
- Legacy TWIST2 and removed policy/task variants remain unsupported

### Training Task
Supported training tasks are `Tracking-Flat-G1-VelCmdHistory` and `Tracking-Flat-G1-MotionTrackingDeploy`.

- VelCmdHistory keeps the current production TemporalCNN path
- MotionTrackingDeploy is a single-stage PPO task aligned to the deployed sim2real policy semantics; no teacher-student or multi-stage pipeline
- Training env uses `sampling_mode="uniform"`
- Playback/benchmark use `play=True`, which switches motion sampling to `start`
- VelCmdHistory defaults to `window_steps=[0]`
- MotionTrackingDeploy uses the deployed future/history reference window and MLP actor/critic
- `save_onnx.py` exports either dual-input TemporalCNN ONNX or single-input deploy MLP ONNX based on checkpoint contents

### Dataset Pipeline
- Dataset build spec supports a `preprocess` section for root-xy normalization, ground alignment, and basic clip filtering
- Dataset build spec supports `window.reference_steps`; merged `train.npz` / `val.npz` now store `window_steps`, `clip_sample_starts`, and `clip_sample_ends`
- `MotionLib` samples only valid center frames for the configured `window_steps`; `window_steps=[0]` preserves the current production behavior

Quick reference:

```bash
python train_mimic/scripts/data/build_dataset.py --spec train_mimic/configs/datasets/twist2_full.yaml
python train_mimic/scripts/train.py --motion_file data/datasets/twist2_full/train.npz
python train_mimic/scripts/train.py --task Tracking-Flat-G1-MotionTrackingDeploy --motion_file data/datasets/twist2_full/train.npz
python train_mimic/scripts/save_onnx.py --checkpoint logs/rsl_rl/g1_tracking_velcmd_history/<run>/model_30000.pt --output policy.onnx --history_length 10
python train_mimic/scripts/save_onnx.py --checkpoint logs/rsl_rl/g1_tracking_motion_tracking_deploy/<run>/model_30000.pt --output policy.onnx
python scripts/run_sim.py --config-name motion_tracking controller.policy_path=policy.onnx input.bvh_file=data/lafan1/dance1_subject2.bvh
python scripts/run_sim2real.py --config-name motion_tracking_sim2real controller.policy_path=policy.onnx
```

### GMR Retargeting
- Self-contained in `teleopit/retargeting/gmr/` with all assets
- Supports `lafan1` BVH (22 joints, 30fps, centimeters)
- Supports `hc_mocap` BVH (50 joints, 60fps downsampled to 30fps, meters)
- `lafan1-resolved` still needs an adapter layer and remains unsupported

### IK Offset Calibration
For each `(robot_body, human_bone)` pair, IK config stores a quaternion offset `R_offset` (`w,x,y,z`, scalar-first):

```
R_result = R_human * R_offset
R_offset = R_human_tpose^{-1} * R_robot_tpose
```

Critical note: align robot root orientation to the BVH human forward direction before computing `R_robot_tpose`. For `hc_mocap`, G1 default faces `+X` while the BVH human faces `-Y` (`Z-up`), so the robot root must receive a `-90°` Z rotation first.

`scripts/compute_ik_offsets.py` can print or write calibrated offsets.

## Development

### Runtime Validation Policy
- Fail fast for logical mismatches such as observation definition vs. ONNX signature mismatch
- Do not silently pad, trim, clip, or replace invalid data/config to "make it run"
- Error messages should identify the mismatched components and the direct fix path

### Commit Policy
- Do not auto-commit changes
- Use the default git user as commit author
- After major feature changes, update `AGENTS.md` and `README.md` together with the code

```bash
pip install -e .
pytest tests/ -v
```

## Known Issues

1. `lafan1-resolved` retargeting is still broken because it uses a different BVH skeleton layout.
2. `g1_mocap_29dof.xml` still has `ctrlrange="-1 1"`; never use it for sim2sim.
