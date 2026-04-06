---
sidebar_position: 2
---

# Config Reference

Complete reference for all configurable fields.

## Top-Level Fields

| Field | Description | Default |
|-------|-------------|---------|
| `policy_hz` | Policy inference frequency | `50` |
| `pd_hz` | PD control frequency | `1000` |
| `viewers` | Viewer set: `mocap`, `retarget`, `sim2sim`, `all`, `none` | `sim2sim` |
| `realtime` | Rate-limit to wall clock | `true` |
| `num_steps` | Number of steps; `0` = infinite | `0` |
| `transition_duration` | Smooth transition time (seconds) from current pose to retarget command | - |
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
| `robot.obs_builder` | Observation builder type | `mjlab` |

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
| `input.provider` | `bvh` |
| `input.bvh_file` | **Required.** Path to BVH file |
| `input.bvh_format` | `lafan1` / `hc_mocap` |
| `input.human_format` | Human skeleton format |

### Pico 4

| Field | Description |
|-------|-------------|
| `input.provider` | `pico4` |
| `input.pico4_timeout` | Wait timeout in seconds |
| `input.pause_button` | Button for pause/resume |

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

## Critical: `default_dof_pos`

The RL policy outputs action **offsets** relative to the default standing pose, not absolute joint angles:

```text
target_dof_pos = clip(action, low, high) * action_scale + default_dof_pos
```

Therefore:
- `default_dof_pos` must align with `robot.default_angles`
- Missing this offset causes the robot to fall immediately

`TeleopPipeline` automatically passes `robot.default_angles` to the controller's `default_dof_pos` during initialization. Understanding this chain is important when writing custom entry points or tests.
