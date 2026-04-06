---
sidebar_position: 1
---

# 离线 Sim2Sim

在 MuJoCo 仿真环境中，使用 BVH 动捕文件驱动 RL 策略进行全身运动复现。

## 基本播放

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

### 使用 hc_mocap 格式

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/hc_mocap/walk.bvh \
    input.bvh_format=hc_mocap
```

## 键盘交互重播

为离线 BVH 播放启用键盘交互控制：

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    playback.keyboard.enabled=true
```

| 按键 | 功能 |
|------|------|
| `Space` / `P` | 暂停 / 继续 |
| `R` | 从头重播 |
| `Q` | 停止 |

其他可选参数：

```bash
# 动作播放结束后自动暂停
playback.pause_on_end=true

# 限制仿真步数（0 = 无限）
num_steps=300

# 按真实时间速率播放（即使无 Viewer 窗口也生效）
realtime=true
```

## Viewer 模式

Viewer 以独立子进程运行。使用 shell 引号传递列表参数。

```bash
viewers=sim2sim          # 默认模式
viewers=all              # mocap + retarget + sim2sim 三视图
viewers=none             # 无头模式（不显示窗口）
'viewers=[retarget,sim2sim]'  # 自定义组合
```

:::note
当所有 Viewer 窗口被关闭后，仿真会自动结束。
:::

## 录制

将仿真数据录制为 HDF5 文件：

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    +record=true \
    recording.output_path=outputs/session.h5
```

录制包含以下字段：`joint_pos`、`joint_vel`、`mimic_obs`、`action`、`target_dof_pos`、`torque`、`timestamp`。

## 离线渲染

在无头模式下将仿真渲染为视频：

```bash
MUJOCO_GL=egl python scripts/render/render_sim.py \
    --bvh data/sample_bvh/aiming1_subject1.bvh \
    --policy track.onnx
```

使用 hc_mocap 格式时：

```bash
MUJOCO_GL=egl python scripts/render/render_sim.py \
    --bvh data/hc_mocap/wander.bvh \
    --format hc_mocap \
    --policy track.onnx
```

渲染管线输出三个视角（动捕输入、重定向、sim2sim），均通过 MuJoCo 渲染。
