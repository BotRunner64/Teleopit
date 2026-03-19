# AGENTS.md

## Project Overview

Teleopit is a lightweight, extensible, self-contained humanoid robot whole-body teleoperation framework. It integrates GMR (General Motion Retargeting) and supports train_mimic-exported ONNX RL policy inference.

Language: Python 3.10+
Package: `teleopit` (installed via `pip install -e .`)
Config: Hydra/OmegaConf YAML files in `teleopit/configs/`

## Architecture

```
InputProvider (BVH file / UDP realtime / Pico4 VR) → Retargeter (GMR) → ObservationBuilder (166D VelCmdHistory) → Controller (dual-input ONNX RL) → Robot (MuJoCo + PD / Unitree SDK)
```

Module-internal isolation: all modules run in-process and communicate via `InProcessBus` (zero-copy). Core interfaces are defined as `typing.Protocol` in `teleopit/interfaces.py`.

## Supported Surface

- Only supported training task: `Tracking-Flat-G1-VelCmdHistory`
- Only supported inference observation: 166D VelCmdHistory
- Only supported ONNX signature: dual inputs `obs` and `obs_history`
- Adaptive sampling, old no-state-estimation MLP tasks, and other legacy task variants are removed

## Directory Structure

```
teleopit/                 # Core inference package
├── interfaces.py         # Protocol definitions: Robot, Controller, InputProvider, Retargeter, etc.
├── pipeline.py           # TeleopPipeline — thin sim runtime facade
├── runtime/              # Shared runtime assembly: config/path resolution, factories, CLI helpers
├── bus/                  # InProcessBus message pub/sub
├── configs/              # Hydra YAML configs
│   ├── default.yaml      # Offline sim top-level config: viewers, policy_hz, pd_hz
│   ├── online.yaml       # Online sim2sim top-level config: realtime=true, num_steps=0
│   ├── robot/g1.yaml     # G1 robot: XML path, PD gains, default angles, action dims
│   ├── controller/rl_policy.yaml
│   ├── input/bvh.yaml    # Offline BVH file input
│   └── input/udp_bvh.yaml # UDP realtime BVH input (reference_bvh, port, timeout)
├── controllers/
│   ├── rl_policy.py      # RLPolicyController — 166D dual-input ONNX inference only
│   └── observation.py    # VelCmdObservationBuilder — only supported 166D observation path
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
├── run_sim2real.py       # G1 sim2real control; supports Pico4 via --config-name pico4_sim2real
├── send_bvh_udp.py       # UDP BVH test sender
├── render_sim.py         # Render single BVH → 3 videos (bvh skeleton, retarget, sim2sim)
├── compute_ik_offsets.py # Compute IK quaternion offsets for new BVH formats
└── setup_pico4.sh        # Pico4 environment setup helper
train_mimic/              # Training package
├── app.py                # Shared app helpers for train/play/benchmark
├── tasks/tracking/config/
│   ├── constants.py      # Single-task public constants
│   ├── registry.py       # Registers only Tracking-Flat-G1-VelCmdHistory
│   ├── env.py            # VelCmdHistory env builder
│   └── rl.py             # VelCmdHistory TemporalCNN PPO runner cfg
├── tasks/tracking/rl/
│   ├── runner.py         # ONNX export wrapper for policy + motion labels
│   ├── conv1d_encoder.py # 1-D CNN encoder for temporal history groups
│   └── temporal_cnn_model.py # TemporalCNN actor/critic model
└── scripts/
    ├── train.py          # Training entry point
    ├── play.py           # Checkpoint playback
    ├── benchmark.py      # Policy evaluation with tracking errors
    └── save_onnx.py      # Export VelCmdHistory TemporalCNN ONNX
```

## Key Technical Details

### Sim2Sim Pipeline
- Policy runs at 50Hz, PD control at 1000Hz (`decimation=20`, `sim_dt=0.001`)
- Action flow: `compute_action()` returns raw action → `get_target_dof_pos()` applies clip `[-10, 10]`, scale `0.5`, and `default_dof_pos`
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

### Pico4 Realtime Input
- `Pico4InputProvider` reads realtime body tracking from `xrobotoolkit_sdk`
- Bone naming follows `xrobot_to_g1.json`
- The provider applies an input-space transform to match the current retarget config
- Do not hardcode that transform as a public coordinate-system contract; validate against actual retarget/sim2sim behavior when SDK or firmware changes

### SimulationLoop Runtime Behavior
- `realtime=true` enforces wall-clock pacing even without a viewer
- `num_steps=0` means infinite loop (`max_steps = 2**63`)
- `KeyboardInterrupt` is handled for clean shutdown
- BVH frame alignment is time-based: `bvh_idx = int(policy_time × input_fps)`

### Inference Observation
The only supported observation is **166D VelCmdHistory**:

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

Runtime constraints:
- `VelCmdObservationBuilder` is the only public observation builder
- `RLPolicyController` only accepts dual-input ONNX with `obs_history`
- Both current observation and history observation dimensions must be exactly `166`
- Startup validates the observation definition against the ONNX signature and raises immediately on mismatch
- Legacy TWIST2, non-VelCmdHistory, and single-input policy paths are removed

### Training Task
The only supported training task is `Tracking-Flat-G1-VelCmdHistory`.

- Training env uses `sampling_mode="uniform"`
- Playback/benchmark use `play=True`, which switches motion sampling to `start`
- Actor/critic both use `TemporalCNNModel`
- Exported ONNX always has dual inputs: `obs` and `obs_history`
- Adaptive sampling is removed; command sampling now supports only `uniform` and `start`

Quick reference:

```bash
python train_mimic/scripts/data/build_dataset.py --spec train_mimic/configs/datasets/twist2_full.yaml
python train_mimic/scripts/train.py --motion_file data/datasets/twist2_full/train.npz
python train_mimic/scripts/save_onnx.py --checkpoint logs/rsl_rl/g1_tracking_velcmd_history/<run>/model_30000.pt --output policy.onnx --history_length 10
python train_mimic/scripts/benchmark.py --checkpoint <path> --motion_file data/datasets/twist2_full/val.npz --num_envs 1
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
