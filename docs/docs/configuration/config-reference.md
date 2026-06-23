---
sidebar_position: 2
---

# Config Reference

Complete reference for all configurable fields.

## Top-Level Fields

| Field | Description | Default |
|-------|-------------|---------|
| `policy_hz` | Policy inference frequency | `50` |
| `pd_hz` | PD control frequency (simulation only) | `200` |
| `viewers` | Viewer set: `mocap`, `retarget`, `sim2sim`, `camera`, `all`, `none`. `all` opens `mocap`, `retarget`, and `sim2sim`; add `camera` explicitly. | `sim2sim` |
| `realtime` | Rate-limit to wall clock | `false` |
| `num_steps` | Number of steps; `0` = infinite | `0` |
| `keyboard.enabled` | Enable realtime keyboard mode control for sim2sim | `false` |
| `playback.pause_on_end` | Pause at last frame when offline motion ends | `false` |
| `playback.keyboard.enabled` | Enable keyboard control for offline playback | `false` |

## Robot

| Field | Description | Default |
|-------|-------------|---------|
| `robot.num_actions` | Joint action dimension | `29` |
| `robot.xml_path` | MuJoCo XML path | - |
| `d435i_rgb` | Fixed RGB camera in the G1 MJCF; use `viewers=[sim2sim,camera]` to display it | - |
| `robot.kps` / `robot.kds` | PD gains | - |
| `robot.default_angles` | Default standing pose | - |
| `robot.torque_limits` | Joint torque limits | - |

## Controller

| Field | Description | Default |
|-------|-------------|---------|
| `controller.policy_path` | **Required.** Path to ONNX policy file | - |
| `controller.device` | Inference device: `cpu` / `auto` / `cuda:N` | `cpu` |
| `controller.action_scale` | Action scaling factor | - |
| `controller.clip_range` | Action clipping range | - |
| `controller.default_dof_pos` | Joint angle offset base | - |

## Input

### Offline BVH

| Field | Description |
|-------|-------------|
| `input.bvh_file` | **Required.** Path to BVH file |
| `input.bvh_format` | `lafan1` / `hc_mocap` |
| `input.human_format` | Human skeleton format |

> BVH input does not set `input.provider` — it is inferred from the config group name.

### Pico 4

| Field | Description | Default |
|-------|-------------|---------|
| `input.provider` | `pico4` | `pico4` |
| `input.human_format` | Retarget skeleton format | `pico_bridge` |
| `input.pico4_timeout` | Wait timeout in seconds | `60` |
| `input.pico4_buffer_size` | Frame buffer size | `60` |
| `input.pause_button` | Button for pause/resume | `A` |
| `input.pause_debounce_s` | Debounce time for pause button | `0.25` |
| `input.arms_button` | Button for Pico `MOCAP` / `ARMS` toggle | `B` |
| `input.arms_debounce_s` | Debounce time for arms-mode button | `0.25` |
| `input.bridge_host` | Teleopit host receiver bind host | `0.0.0.0` |
| `input.bridge_port` | Teleopit host receiver TCP/UDP port | `63901` |
| `input.bridge_discovery` | Enable pico-bridge discovery advertising | `true` |
| `input.bridge_advertise_ip` | Optional advertised host IP override | `null` |
| `input.bridge_start_timeout` | Timeout while starting the bridge | `10.0` |
| `input.bridge_history_size` | Pico frame history retained by the bridge | `120` |
| `input.video.enabled` | Stream host camera preview back to Pico through pico-bridge 0.2.1 | `false` |
| `input.video.source` | Video source: `mujoco`, `realsense`, or `test-pattern` | `null` |
| `input.video.width` / `height` / `fps` | Video capture/render settings | `1280` / `720` / `30` |
| `input.video.device` | Optional RealSense serial | `null` |
| `input.video.fail_on_error` | Fail startup instead of disabling video on error | `false` |

### Realtime

| Field | Description |
|-------|-------------|
| `retarget_buffer_enabled` | Enable retarget buffering |
| `retarget_buffer_window_s` | Buffer window size |
| `retarget_buffer_delay_s` | Buffer delay |
| `reference_steps` | Reference window steps |
| `realtime_buffer_warmup_steps` | Warmup before playback |
| `reference_velocity_smoothing_alpha` | Velocity smoothing |
| `reference_anchor_velocity_smoothing_alpha` | Anchor velocity smoothing |

## Sim2Real

Fields used by sim2real configs (`sim2real.yaml`, `pico4_sim2real.yaml`).

Sim2real defaults to `viewers=none`. Set `viewers=retarget` to open an optional
MuJoCo window showing the retargeted reference; `sim2sim`, `mocap`, `camera`,
and `all` are simulation-only viewer modes.

### Safety

| Field | Description | Default |
|-------|-------------|---------|
| `startup_ramp_duration` | Kp ramp duration after entering `STANDING`; gradually increases PD gains without changing policy targets | `2.0` |
| `joint_vel_limit` | Joint velocity limit (rad/s); triggers emergency damping if exceeded | `10.0` |
| `mocap_switch.check_frames` | Consecutive valid frames required before switching to MOCAP | `10` |
| `arm_mocap.controlled_joint_indices` | G1 joints driven by live retargeting in Pico `ARMS` mode | `[15..28]` |

### Real Robot

| Field | Description | Default |
|-------|-------------|---------|
| `real_robot.network_interface` | Network interface for Unitree DDS communication. For wired PC-to-G1 control, find the cable interface with `ifconfig` and set that name, for example `enp130s0`; for onboard robot execution, `eth0` is usually correct. | `eth0` |
| `real_robot.kp_real` | Real-robot proportional gains (per joint) | - |
| `real_robot.kd_real` | Real-robot derivative gains (per joint) | - |
| `real_robot.kd_damping` | Damping mode kd | `8.0` |
| `real_robot.control_mode` | Ankle control mode (`PR` = Pitch-Roll) | `PR` |
| `real_robot.joint_pos_lower` | Joint position lower limits (rad) | - |
| `real_robot.joint_pos_upper` | Joint position upper limits (rad) | - |

### Pause/Resume (Pico sim2real)

Realtime Pico resume re-centers heading and ground-plane position before tracking continues. Operators should keep still and stay as close as practical to the paused pose to reduce sudden reference changes.

### Dexterous Hand (Pico sim2real)

`hands.enabled=true` requires `input.provider=pico4` plus local editable
installs of `third_party/linkerhand-python-sdk` and `third_party/somehand`.
Control is active in `MOCAP` and `ARMS`; inactive modes send the open pose.
`gripper` supports `linkerhand_l6` and `linkerhand_o6` by interpolating Pico
trigger input between the configured open and close poses. `vr_hand_pose` is
L6-only: missing hand pose holds the last command for that side, L6 speed is
set to the maximum, and Teleopit converts Pico hand state to 21 landmarks before
calling somehand 0.2.0 through `somehand.api` only.

| Field | Description | Default |
|-------|-------------|---------|
| `hands.enabled` | Enable optional hand worker | `false` |
| `hands.driver` | Hand driver plugin: `linkerhand_l6` or `linkerhand_o6` | `linkerhand_l6` |
| `hands.mode` | `gripper` or `vr_hand_pose` | `gripper` |
| `hands.sides` | Controlled sides | `[left, right]` |
| `hands.rate_hz` | Maximum gripper command rate in Hz | `30.0` |
| `hands.frame_timeout_s` | Controller or hand-pose staleness threshold | `0.3` |
| `hands.linkerhand_l6.left_can` / `right_can` | CAN channels for each hand | `can0` / `can1` |
| `hands.linkerhand_l6.speed` | L6 speed used by `gripper`; `vr_hand_pose` overrides this to maximum speed | see config |
| `hands.linkerhand_l6.open_pose` / `close_pose` | Six-value L6 open/closed poses | see config |
| `hands.linkerhand_o6.left_can` / `right_can` | CAN channels for each O6 hand | `can0` / `can1` |
| `hands.linkerhand_o6.speed` | O6 speed used by `gripper` | see config |
| `hands.linkerhand_o6.open_pose` / `close_pose` | Six-value O6 open/closed poses | see config |
| `hands.somehand.config_path` | Official somehand 0.2.0 bi-hand L6 config used by `vr_hand_pose` | see config |
| `hands.somehand.rate_hz` | Low-latency `vr_hand_pose` command rate in Hz | `60.0` |
| `hands.somehand.max_iterations` | somehand solver iteration cap for `vr_hand_pose` | `12` |
| `hands.somehand.temporal_filter_alpha` | somehand input landmark smoothing alpha; `1.0` disables smoothing delay | `1.0` |
| `hands.somehand.output_alpha` | somehand qpos output smoothing alpha; `1.0` disables smoothing delay | `1.0` |

### HDF5 Recording (Pico sim2real)

`recording.enabled=true` is supported only with `input.provider=pico4`,
`input.video.enabled=true`, `input.video.source=realsense`, and an interactive
terminal. The recorder is manual: `R` starts an episode, `S` saves the active
episode, `D` discards the active episode, and `Q` shuts down. `STANDING`,
`MOCAP`, `ARMS`, and paused mocap can be recorded.

`sim2real_record.yaml` enables both recording and the required RealSense
`input.video` path. Recording does not open a second camera; it consumes the
same frames produced by `pico_input`.

| Field | Description | Default |
|-------|-------------|---------|
| `recording.enabled` | Enable manual HDF5 recording | `false` |
| `recording.output_dir` | Dataset root directory | `data/recordings/sim2real_hdf5` |
| `recording.task` | Task string stored with frames | `demo` |
| `recording.fps` | Recording/video clock rate | `30` |
| `recording.min_episode_seconds` | Discard saved episodes shorter than this duration | `1.0` |
| `recording.record_modes` | Modes that allow recording start and frame writes | `[standing, mocap, arms, pause]` |
| `recording.camera.key` | RGB image dataset key | `observation.images.d435i_rgb` |
| `recording.camera.width` / `height` / `fps` | RealSense RGB capture settings | `640` / `480` / `30` |
| `recording.camera.device` | Optional RealSense serial | `null` |
| `recording.video.codec` / `quality` / `pixelformat` | MP4 sidecar encoder settings | `libx264` / `8` / `yuv420p` |

Camera failure behavior is controlled by `input.video.fail_on_error`.

Each saved episode has one `.h5` file under `recording.output_dir/episodes/`
and one compressed MP4 sidecar under
`recording.output_dir/videos/<camera_key>/`. The HDF5 episode stores
`frame_index` and `timestamp` arrays, plus `video_path`, `video_fps`, and
`video_frames` root attributes for synchronization. Raw RGB image datasets are
not written.

HDF5 datasets:

```text
frame_index                    int64[N]
timestamp                      float64[N]
observation.state              float32[68]
observation.mode               float32[1]
action                         float32[36]
action.hand                    float32[12]
```

The root attributes include the Teleopit HDF5 recording format, schema version,
task, fps, frame count, and video sync metadata.

`observation.state` is ordered as `joint_pos(29)`, `joint_vel(29)`,
`base_quat_wxyz(4)`, `base_ang_vel(3)`, and `projected_gravity(3)`.
`observation.mode` is a numeric categorical: `standing=0`, `mocap=1`,
`arms=2`, and `pause=3`. `action` is the current reference qpos:
`root_pos(3) + root_quat_wxyz(4) + joint_pos(29)`.
`action.hand` is the latest LinkerHand command from the hand worker:
`left_pose(6) + right_pose(6)`, using the SDK's 0-255 pose values.

## Critical: `default_dof_pos`

The RL policy outputs action **offsets** relative to the default standing pose, not absolute joint angles:

```text
target_dof_pos = clip(action, low, high) * action_scale + default_dof_pos
```

Therefore:
- `default_dof_pos` must align with `robot.default_angles`
- Missing this offset causes the robot to fall immediately

`TeleopPipeline` automatically passes `robot.default_angles` to the controller's `default_dof_pos` during initialization. Understanding this chain is important when writing custom entry points or tests.
