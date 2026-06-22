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
| `arms_button` | str | `B` | Pico 中用于切换 `MOCAP` / `ARMS` 的按钮 |
| `arms_debounce_s` | float | `0.25` | 双臂模式按钮防抖时间 |
| `bridge_host` | str | `0.0.0.0` | Teleopit host receiver 绑定地址 |
| `bridge_port` | int | `63901` | Teleopit host receiver TCP/UDP 端口 |
| `bridge_discovery` | bool | `true` | 是否启用 pico-bridge 发现广播 |
| `bridge_advertise_ip` | str/null | `null` | 可选的 host 广播 IP 覆盖 |
| `bridge_start_timeout` | float | `10.0` | 启动 bridge 的超时时间 |
| `bridge_history_size` | int | `120` | bridge 保留的 Pico 帧历史长度 |
| `video.enabled` | bool | `false` | 通过 pico-bridge 0.2.1 将 host 相机预览发送回 Pico |
| `video.source` | str/null | `null` | 视频源：`mujoco`、`realsense` 或 `test-pattern` |
| `video.width` / `height` / `fps` | int | `1280` / `720` / `30` | 视频采集/渲染设置 |
| `video.device` | str/null | `null` | 可选的 RealSense 序列号 |
| `video.fail_on_error` | bool | `false` | 视频失败时是否让启动失败，而不是关闭视频后继续 |

## Realtime 字段

实时模式相关字段，仅在 `realtime=true` 时生效。

| 字段 | 说明 |
|---|---|
| `retarget_buffer_enabled` | 是否启用重定向缓冲 |
| `retarget_buffer_window_s` | 缓冲窗口大小 |
| `retarget_buffer_delay_s` | 缓冲延迟 |
| `reference_steps` | 参考轨迹窗口步数 |
| `realtime_buffer_warmup_steps` | 播放前预热帧数 |
| `reference_velocity_smoothing_alpha` | 速度平滑系数 |
| `reference_anchor_velocity_smoothing_alpha` | 锚点速度平滑系数 |

## Sim2Real 字段

以下字段用于 sim2real 配置（`sim2real.yaml`、`pico4_sim2real.yaml`）。

sim2real 默认使用 `viewers=none`。设置 `viewers=retarget` 可打开一个可选的
MuJoCo 窗口显示重定向参考；`sim2sim`、`mocap`、`camera` 和 `all`
仅用于仿真 viewer。

### 安全相关

| 字段 | 说明 | 默认值 |
|---|---|---|
| `startup_ramp_duration` | 进入 `STANDING` 后的 Kp ramp 时长；逐步提高 PD 增益，不改变 policy target | `2.0` |
| `joint_vel_limit` | 关节速度限制（rad/s），超过时触发急停 | `10.0` |
| `mocap_switch.check_frames` | 切换到 MOCAP 前所需的连续有效帧数 | `10` |
| `arm_mocap.controlled_joint_indices` | Pico `ARMS` 模式下由实时 retargeting 驱动的 G1 关节 | `[15..28]` |

### 真机 SDK

| 字段 | 说明 | 默认值 |
|---|---|---|
| `real_robot.network_interface` | Unitree DDS 通信网络接口。PC 通过网线连接 G1 控制时，用 `ifconfig` 找到这根网线对应的接口名并填写，例如 `enp130s0`；在机器人 onboard 计算机上运行时通常使用 `eth0` | `eth0` |
| `real_robot.kp_real` | 真机比例增益（各关节） | — |
| `real_robot.kd_real` | 真机微分增益（各关节） | — |
| `real_robot.kd_damping` | 阻尼模式 kd | `8.0` |
| `real_robot.control_mode` | 踝关节控制模式（`PR` = Pitch-Roll） | `PR` |
| `real_robot.joint_pos_lower` | 关节位置下限（rad） | — |
| `real_robot.joint_pos_upper` | 关节位置上限（rad） | — |

### 暂停/恢复（Pico sim2real）

实时 Pico 恢复追踪时会先重新居中航向和地面平面位置。操作者应保持静止，并尽量贴近暂停时的姿态，以减少参考突变。

### 灵巧手（Pico sim2real）

`hands.enabled=true` 要求 `input.provider=pico4`，并以本地 editable 方式安装
`third_party/linkerhand-python-sdk` 和 `third_party/somehand`。控制在 `MOCAP`
和 `ARMS` 中生效；非活动模式会发送张开姿态。
`gripper` 支持 `linkerhand_l6` 和 `linkerhand_o6`，会用 Pico trigger 在配置的张开和闭合姿态之间插值。
`vr_hand_pose` 只支持 L6：手部 pose 消失时，对应侧会保持上一条命令；L6 速度会设为最大值；
Teleopit 会先将 Pico 手部状态转成 21 个 landmarks，再只通过 somehand 0.2.0 公开的 `somehand.api` 调用。

| 字段 | 说明 | 默认值 |
|---|---|---|
| `hands.enabled` | 启用可选手部运行时 | `false` |
| `hands.mode` | `gripper` 或 `vr_hand_pose` | `gripper` |
| `hands.driver` | 手部设备驱动：`linkerhand_l6` 或 `linkerhand_o6` | `linkerhand_l6` |
| `hands.sides` | 控制侧 | `[left, right]` |
| `hands.rate_hz` | gripper 最大命令频率（Hz） | `30.0` |
| `hands.frame_timeout_s` | 手柄或手部 pose 过期阈值 | `0.3` |
| `hands.linkerhand_l6.left_can` / `right_can` | 左右手 CAN 通道 | `can0` / `can1` |
| `hands.linkerhand_l6.speed` | `gripper` 使用的 L6 速度；`vr_hand_pose` 会覆盖为最大速度 | 见配置 |
| `hands.linkerhand_l6.deadman_threshold` | 启用单侧控制所需的最小 grip 值 | `0.5` |
| `hands.linkerhand_l6.trigger_deadzone` | trigger 两端死区 | `0.05` |
| `hands.linkerhand_l6.open_pose` / `close_pose` | L6 的 6 维张开/闭合姿态 | 见配置 |
| `hands.linkerhand_o6.left_can` / `right_can` | 左右 O6 手 CAN 通道 | `can0` / `can1` |
| `hands.linkerhand_o6.speed` | `gripper` 使用的 O6 速度 | 见配置 |
| `hands.linkerhand_o6.open_pose` / `close_pose` | O6 的 6 维张开/闭合姿态 | 见配置 |
| `hands.somehand.config_path` | `vr_hand_pose` 使用的 somehand 双手 L6 配置 | 见配置 |
| `hands.somehand.rate_hz` | 低延时 `vr_hand_pose` 命令频率（Hz） | `60.0` |
| `hands.somehand.max_iterations` | `vr_hand_pose` 的 somehand solver 迭代上限 | `12` |
| `hands.somehand.temporal_filter_alpha` | somehand 输入 landmarks 平滑 alpha；`1.0` 表示关闭平滑延时 | `1.0` |
| `hands.somehand.output_alpha` | somehand qpos 输出平滑 alpha；`1.0` 表示关闭平滑延时 | `1.0` |

### LeRobot 录制（Pico sim2real）

`recording.enabled=true` 只支持 `input.provider=pico4`、
`input.video.enabled=true`、`input.video.source=realsense`，并且需要交互式终端。
录制是手动控制：`R` 开始 episode，`S` 保存当前 episode，`D` 丢弃当前 episode，
`Q` 关闭。可以录制 `STANDING`、`MOCAP`、`ARMS` 和暂停状态的 mocap。

`sim2real_record.yaml` 会同时启用录制和必需的 RealSense `input.video`
路径。录制不会打开第二路相机，而是消费 `pico_input` 已经产生的同一批帧。

| 字段 | 说明 | 默认值 |
|---|---|---|
| `recording.enabled` | 启用手动 LeRobot v3 录制 | `false` |
| `recording.output_dir` | 数据集根目录 | `data/lerobot` |
| `recording.repo_id` / `dataset_name` | LeRobot 数据集标识 | `null` |
| `recording.task` | 写入 frame 的任务字符串 | `demo` |
| `recording.fps` | 录制/视频主时钟频率 | `30` |
| `recording.min_episode_seconds` | 保存时短于该时长的 episode 会被丢弃 | `1.0` |
| `recording.record_modes` | 允许开始录制和写帧的模式 | `[standing, mocap, arms, pause]` |
| `recording.camera.key` | LeRobot 视频 feature key | `observation.images.d435i_rgb` |
| `recording.camera.width` / `height` / `fps` | RealSense RGB 采集设置 | `640` / `480` / `30` |
| `recording.camera.device` | 可选 RealSense 序列号 | `null` |

相机失败时的行为由 `input.video.fail_on_error` 控制。

LeRobot features：

```text
observation.images.d435i_rgb   video [480,640,3] uint8
observation.state              float32[68]
observation.mode               float32[1]
action                         float32[36]
```

`observation.state` 的顺序是 `joint_pos(29)`、`joint_vel(29)`、
`base_quat_wxyz(4)`、`base_ang_vel(3)` 和 `projected_gravity(3)`。
`observation.mode` 是数值类别：`standing=0`、`mocap=1`、
`arms=2`、`pause=3`。`action` 是当前 reference qpos：
`root_pos(3) + root_quat_wxyz(4) + joint_pos(29)`。
