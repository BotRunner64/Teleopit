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
| `mocap_switch.max_position_value` | Position sanity threshold in meters | `5.0` |

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

`dexterous_hand.mode=gripper` or `dexterous_hand.mode=vr_hand_pose` requires
`input.provider=pico4` and the optional `dexhand` extra. Control is active only
in `MOCAP`; inactive modes send the open pose. In `vr_hand_pose`, missing hand
pose holds the last command for that side. `gripper` uses the configured
`dexterous_hand.speed`; `vr_hand_pose` always sets LinkerHand L6 speed to the
maximum.

| Field | Description | Default |
|-------|-------------|---------|
| `dexterous_hand.mode` | `off`, `gripper`, or `vr_hand_pose` | `off` |
| `dexterous_hand.hand_type` | Controlled side: `left`, `right`, or `both`; `vr_hand_pose` requires `both` | `both` |
| `dexterous_hand.left_can` / `right_can` | CAN channels for each hand | `can0` / `can1` |
| `dexterous_hand.rate` | Maximum command rate in Hz | `30.0` |
| `dexterous_hand.frame_timeout` | Gripper controller timeout, or VR hand-pose staleness threshold | `0.3` |
| `dexterous_hand.speed` | L6 speed used by `gripper`; `vr_hand_pose` overrides this to maximum speed | see config |
| `dexterous_hand.deadman_threshold` | Minimum grip value required to enable a side | `0.5` |
| `dexterous_hand.trigger_deadzone` | Trigger deadzone at both ends | `0.05` |
| `dexterous_hand.open_pose` / `close_pose` | Six-value L6 open/closed poses | see config |
| `dexterous_hand.somehand.config_path` | somehand bi-hand L6 config used by `vr_hand_pose` | see config |

## Critical: `default_dof_pos`

The RL policy outputs action **offsets** relative to the default standing pose, not absolute joint angles:

```text
target_dof_pos = clip(action, low, high) * action_scale + default_dof_pos
```

Therefore:
- `default_dof_pos` must align with `robot.default_angles`
- Missing this offset causes the robot to fall immediately

`TeleopPipeline` automatically passes `robot.default_angles` to the controller's `default_dof_pos` during initialization. Understanding this chain is important when writing custom entry points or tests.
