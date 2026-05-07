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
| `viewers` | Viewer set: `mocap`, `retarget`, `sim2sim`, `all`, `none` | `sim2sim` |
| `realtime` | Rate-limit to wall clock | `false` |
| `num_steps` | Number of steps; `0` = infinite | `0` |
| `transition_duration` | Smooth transition time (seconds) from current pose to retarget command | - |
| `keyboard.enabled` | Enable realtime keyboard mode control for sim2sim | `false` |
| `playback.pause_on_end` | Pause at last frame when offline motion ends | `false` |
| `playback.keyboard.enabled` | Enable keyboard control for offline playback | `false` |

## Robot

| Field | Description | Default |
|-------|-------------|---------|
| `robot.num_actions` | Joint action dimension | `29` |
| `robot.xml_path` | MuJoCo XML path | - |
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
| `input.bridge_host` | PC receiver bind host | `0.0.0.0` |
| `input.bridge_port` | PC receiver UDP port | `63901` |
| `input.bridge_discovery` | Enable pico-bridge discovery advertising | `true` |
| `input.bridge_advertise_ip` | Optional advertised PC IP override | `null` |
| `input.bridge_video` | Optional pico-bridge video mode | `null` |
| `input.bridge_camera_device` | Optional camera device for bridge video | `null` |
| `input.bridge_start_timeout` | Timeout while starting the bridge | `10.0` |
| `input.bridge_history_size` | Pico frame history retained by the bridge | `120` |

### Realtime

| Field | Description |
|-------|-------------|
| `retarget_buffer_enabled` | Enable retarget buffering |
| `retarget_buffer_window_s` | Buffer window size |
| `retarget_buffer_delay_s` | Buffer delay |
| `reference_steps` | Reference window steps |
| `realtime_buffer_warmup_steps` | Warmup before playback |
| `realtime_buffer_low_watermark_steps` | Low watermark |
| `realtime_buffer_high_watermark_steps` | High watermark |
| `reference_velocity_smoothing_alpha` | Velocity smoothing |
| `reference_anchor_velocity_smoothing_alpha` | Anchor velocity smoothing |

## Sim2Real

Fields used by sim2real configs (`sim2real.yaml`, `pico4_sim2real.yaml`).

### Safety

| Field | Description | Default |
|-------|-------------|---------|
| `startup_ramp_duration` | Seconds to smoothly blend from locked to policy positions | `2.0` |
| `joint_vel_limit` | Joint velocity limit (rad/s); triggers emergency damping if exceeded | `10.0` |
| `mocap_switch.check_frames` | Consecutive valid frames required before switching to MOCAP | `10` |
| `mocap_switch.max_position_value` | Position sanity threshold in meters | `5.0` |

### Real Robot

| Field | Description | Default |
|-------|-------------|---------|
| `real_robot.network_interface` | Network interface for DDS communication | `eth0` |
| `real_robot.kp_real` | Real-robot proportional gains (per joint) | - |
| `real_robot.kd_real` | Real-robot derivative gains (per joint) | - |
| `real_robot.kd_damping` | Damping mode kd | `8.0` |
| `real_robot.control_mode` | Ankle control mode (`PR` = Pitch-Roll) | `PR` |
| `real_robot.joint_pos_lower` | Joint position lower limits (rad) | - |
| `real_robot.joint_pos_upper` | Joint position upper limits (rad) | - |

### Pause/Resume (Pico sim2real)

| Field | Description | Default |
|-------|-------------|---------|
| `pause_resume_transition_duration` | Seconds to blend from paused pose back to live mocap | `1.0` |
| `pause_resume_warmup_steps` | Mocap frames to accumulate before resuming | `2` |
| `pause_reset_alignment_on_resume` | Rebuild yaw/pivot alignment after pause | `true` |

### Realtime Catch-up (Pico sim2real)

| Field | Description | Default |
|-------|-------------|---------|
| `realtime_catchup_enabled` | Enable catch-up when buffer grows too large | `true` |
| `realtime_catchup_trigger_steps` | Buffer depth that triggers catch-up | `6` |
| `realtime_catchup_release_steps` | Buffer depth to release catch-up | `3` |
| `realtime_catchup_target_delay_s` | Target delay for catch-up | `0.04` |
| `reference_qpos_smoothing_alpha` | Joint position smoothing (1.0 = no smoothing) | `0.4` |

## Critical: `default_dof_pos`

The RL policy outputs action **offsets** relative to the default standing pose, not absolute joint angles:

```text
target_dof_pos = clip(action, low, high) * action_scale + default_dof_pos
```

Therefore:
- `default_dof_pos` must align with `robot.default_angles`
- Missing this offset causes the robot to fall immediately

`TeleopPipeline` automatically passes `robot.default_angles` to the controller's `default_dof_pos` during initialization. Understanding this chain is important when writing custom entry points or tests.
