---
sidebar_position: 2
---

# 配置参考

本页列出 Teleopit 所有可配置字段及其含义。

## 顶层字段

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `policy_hz` | int | — | 策略推理频率（Hz） |
| `pd_hz` | int | `200` | PD 控制器频率（Hz，仅仿真），通常高于 `policy_hz` |
| `viewers` | str/list | `sim2sim` | 可视化窗口集合：`mocap`、`retarget`、`sim2sim`、`camera`、`all`、`none`。`all` 打开 `mocap`、`retarget` 和 `sim2sim`；如需相机画面需显式加入 `camera` |
| `realtime` | bool | `false` | 是否启用实时模式（实机部署时需开启） |
| `num_steps` | int | — | 仿真总步数；设为 `-1` 表示无限运行 |
| `transition_duration` | float | — | 从静止姿态过渡到策略控制的时长（秒） |
| `keyboard.enabled` | bool | `false` | 是否启用 sim2sim 实时键盘模式控制 |
| `playback.pause_on_end` | bool | `false` | 回放结束后是否暂停（而非退出） |
| `playback.keyboard.enabled` | bool | `false` | 是否启用键盘控制回放进度 |

## Robot 字段

机器人相关配置位于 `robot/` 子目录。以 `robot/g1.yaml` 为例：

| 字段 | 类型 | 说明 |
|---|---|---|
| `num_actions` | int | 策略输出的动作维度（即受控关节数） |
| `xml_path` | str | MuJoCo MJCF 模型文件路径 |
| `d435i_rgb` | camera | G1 MJCF 中的固定 RGB 相机；配合 `viewers=[sim2sim,camera]` 显示画面 |
| `kps` | list[float] | 各关节的比例增益（P 增益） |
| `kds` | list[float] | 各关节的微分增益（D 增益） |
| `default_angles` | list[float] | 默认关节角度（弧度），也是策略动作的零点 |
| `torque_limits` | list[float] | 各关节的力矩上限 |

## Controller 字段

控制器配置位于 `controller/` 子目录。

| 字段 | 类型 | 说明 |
|---|---|---|
| `policy_path` | str | **必填。** 策略模型文件路径（ONNX 格式） |
| `device` | str | 推理设备，如 `"cpu"` 或 `"cuda:0"` |
| `action_scale` | float | 动作缩放系数 |
| `clip_range` | list[float] | 动作裁剪范围，格式为 `[min, max]` |
| `default_dof_pos` | list[float] | 默认关节位置，用于计算控制目标 |

### 关键说明：`default_dof_pos` 与动作计算

策略输出的 action 是相对于 `default_dof_pos` 的**偏移量**，最终的关节控制目标按如下公式计算：

```
target = clip(action, clip_range) * action_scale + default_dof_pos
```

因此，`default_dof_pos` 决定了策略输出的"零点"。如果该值与训练时使用的不一致，策略的行为将完全偏离预期。

## Input 字段

输入源配置位于 `input/` 子目录，不同输入源的字段各异。

### BVH 输入（`input/bvh.yaml`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `bvh_file` | str | BVH 文件路径 |
| `bvh_format` | str | BVH 骨骼格式标识 |
| `human_format` | str | 人体骨架格式 |

> BVH 输入不设置 `input.provider` — 由配置组名自动推断。

### Pico 4 输入（`input/pico4.yaml`）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `provider` | str | `pico4` | 输入源类型 |
| `human_format` | str | `pico_bridge` | 重定向骨架格式 |
| `pico4_timeout` | float | `60` | 等待设备连接的超时时间（秒） |
| `pico4_buffer_size` | int | `60` | 帧缓冲区大小 |
| `pause_button` | str | `A` | 用于暂停/恢复的手柄按钮名称 |
| `pause_debounce_s` | float | `0.25` | 暂停按钮防抖时间 |
| `bridge_host` | str | `0.0.0.0` | PC receiver 绑定地址 |
| `bridge_port` | int | `63901` | PC receiver UDP 端口 |
| `bridge_discovery` | bool | `true` | 是否启用 pico-bridge 发现广播 |
| `bridge_advertise_ip` | str/null | `null` | 可选的 PC 广播 IP 覆盖 |
| `bridge_video` | str/null | `null` | 可选的 pico-bridge 视频模式 |
| `bridge_camera_device` | str/null | `null` | 可选的视频相机设备 |
| `bridge_start_timeout` | float | `10.0` | 启动 bridge 的超时时间 |
| `bridge_history_size` | int | `120` | bridge 保留的 Pico 帧历史长度 |

## Realtime 字段

实时模式相关字段，仅在 `realtime=true` 时生效。

| 字段 | 说明 |
|---|---|
| `retarget_buffer_enabled` | 是否启用重定向缓冲 |
| `retarget_buffer_window_s` | 缓冲窗口大小 |
| `retarget_buffer_delay_s` | 缓冲延迟 |
| `reference_steps` | 参考轨迹窗口步数 |
| `realtime_buffer_warmup_steps` | 播放前预热帧数 |
| `realtime_buffer_low_watermark_steps` | 低水位线 |
| `realtime_buffer_high_watermark_steps` | 高水位线 |
| `reference_velocity_smoothing_alpha` | 速度平滑系数 |
| `reference_anchor_velocity_smoothing_alpha` | 锚点速度平滑系数 |

## Sim2Real 字段

以下字段用于 sim2real 配置（`sim2real.yaml`、`pico4_sim2real.yaml`）。

### 安全相关

| 字段 | 说明 | 默认值 |
|---|---|---|
| `startup_ramp_duration` | 从锁定位置平滑过渡到策略控制的时长（秒） | `2.0` |
| `joint_vel_limit` | 关节速度限制（rad/s），超过时触发急停 | `10.0` |
| `mocap_switch.check_frames` | 切换到 MOCAP 前所需的连续有效帧数 | `10` |
| `mocap_switch.max_position_value` | 位置合理性阈值（米） | `5.0` |

### 真机 SDK

| 字段 | 说明 | 默认值 |
|---|---|---|
| `real_robot.network_interface` | DDS 通信网络接口 | `eth0` |
| `real_robot.kp_real` | 真机比例增益（各关节） | — |
| `real_robot.kd_real` | 真机微分增益（各关节） | — |
| `real_robot.kd_damping` | 阻尼模式 kd | `8.0` |
| `real_robot.control_mode` | 踝关节控制模式（`PR` = Pitch-Roll） | `PR` |
| `real_robot.joint_pos_lower` | 关节位置下限（rad） | — |
| `real_robot.joint_pos_upper` | 关节位置上限（rad） | — |

### 暂停/恢复（Pico sim2real）

| 字段 | 说明 | 默认值 |
|---|---|---|
| `pause_resume_transition_duration` | 离线回放恢复时的混合时长；实时 Pico 恢复使用实时追踪重新居中 | `1.0` |
| `pause_resume_warmup_steps` | 恢复追踪前采集的实时动捕帧数 | `2` |

实时 Pico 恢复追踪时会先重新居中航向和地面平面位置。操作者应保持静止，并尽量贴近暂停时的姿态，以减少参考突变。

### 实时追赶（Pico sim2real）

| 字段 | 说明 | 默认值 |
|---|---|---|
| `realtime_catchup_enabled` | 缓冲区过大时启用追赶 | `true` |
| `realtime_catchup_trigger_steps` | 触发追赶的缓冲区深度 | `6` |
| `realtime_catchup_release_steps` | 释放追赶的缓冲区深度 | `3` |
| `realtime_catchup_target_delay_s` | 追赶目标延迟 | `0.04` |
| `reference_qpos_smoothing_alpha` | 关节位置平滑系数（1.0 = 无平滑） | `0.4` |
