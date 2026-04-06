---
sidebar_position: 2
---

# 配置参考

本页列出 Teleopit 所有可配置字段及其含义。

## 顶层字段

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `policy_hz` | int | — | 策略推理频率（Hz） |
| `pd_hz` | int | — | PD 控制器频率（Hz），通常高于 `policy_hz` |
| `viewers` | bool | `false` | 是否启用可视化窗口 |
| `realtime` | bool | `false` | 是否启用实时模式（实机部署时需开启） |
| `num_steps` | int | — | 仿真总步数；设为 `-1` 表示无限运行 |
| `transition_duration` | float | — | 从静止姿态过渡到策略控制的时长（秒） |
| `playback.pause_on_end` | bool | `false` | 回放结束后是否暂停（而非退出） |
| `playback.keyboard.enabled` | bool | `false` | 是否启用键盘控制回放进度 |

## Robot 字段

机器人相关配置位于 `robot/` 子目录。以 `robot/g1.yaml` 为例：

| 字段 | 类型 | 说明 |
|---|---|---|
| `num_actions` | int | 策略输出的动作维度（即受控关节数） |
| `xml_path` | str | MuJoCo MJCF 模型文件路径 |
| `kps` | list[float] | 各关节的比例增益（P 增益） |
| `kds` | list[float] | 各关节的微分增益（D 增益） |
| `default_angles` | list[float] | 默认关节角度（弧度），也是策略动作的零点 |
| `torque_limits` | list[float] | 各关节的力矩上限 |
| `obs_builder` | str | 观测构建器的注册名称 |

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
| `provider` | str | 输入源类型，固定为 `"bvh"` |
| `bvh_file` | str | BVH 文件路径 |
| `bvh_format` | str | BVH 骨骼格式标识 |

### Pico 4 输入（`input/pico4.yaml`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `provider` | str | 输入源类型，固定为 `"pico4"` |
| `timeout` | float | 等待设备连接的超时时间（秒） |
| `pause_button` | str | 用于暂停/恢复的手柄按钮名称 |

## Realtime 字段

实时模式相关字段，仅在 `realtime=true` 时生效。

| 字段 | 类型 | 说明 |
|---|---|---|
| `retarget_buffer` | int | 重定向缓冲区大小（帧数） |
| `reference_steps` | int | 参考轨迹的前瞻步数 |
| `watermark` | int | 缓冲区水位线，低于此值时触发填充 |
