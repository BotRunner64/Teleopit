# 推理与运行指南

本文档覆盖离线 sim2sim、离线文件重播、viewer、录制和离线渲染。配置约束与 FAQ 见 [configuration.md](configuration.md)。

## 运行前确认

- 已安装：`pip install -e .`
- 已准备 ONNX policy（见 [README](../README.md) 下载说明）
- 命令行需显式提供 `controller.policy_path=...` 和 `input.bvh_file=...`（离线模式）

## 离线 sim2sim

```bash
python scripts/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/lafan1/dance1_subject2.bvh
```

`hc_mocap` 示例：

```bash
python scripts/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/hc_mocap/walk.bvh \
    input.bvh_format=hc_mocap
```

常用 override：

```bash
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=none
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=all
python scripts/run_sim.py controller.policy_path=policy.onnx 'viewers=[retarget,sim2sim]'
python scripts/run_sim.py controller.policy_path=policy.onnx 'viewers=[mocap,retarget,sim2sim]'
python scripts/run_sim.py \
    controller.policy_path=policy.onnx \
    +num_steps=5000 \
    +record=true
```

## 离线文件键盘重播

```bash
python scripts/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    playback.keyboard.enabled=true
```

常见 override：

```bash
python scripts/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    playback.keyboard.enabled=true \
    playback.pause_on_end=true
python scripts/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    viewers=none
python scripts/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    num_steps=300
```

运行时行为：

- 离线 BVH 直接在进程内采样，不再需要 UDP relay。
- `Space` / `P` 暂停或恢复，`R` 从头重播，`Q` 停止当前 sim2sim 运行。
- `playback.pause_on_end=true` 时动作播完会停在最后一帧等待手动重播。
- `num_steps=0` 表示无限循环。
- `realtime=true` 时即使没有 viewer 也会做 wall-clock 限速。

## Viewer 模式

`viewers` 支持：

- `sim2sim`
- `retarget`
- `mocap`（retargeting 输入骨架，MuJoCo 自定义几何）
- `all`
- `none`

注意事项：

- 多 viewer 运行在独立子进程中。
- Hydra list override 需要 shell 引号，例如 `'viewers=[retarget,sim2sim]'`。
- 当所有活动 viewer 被关闭时，仿真会自动结束。

## 录制

```bash
python scripts/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/lafan1/dance1_subject2.bvh \
    +record=true \
    recording.output_path=outputs/session.h5
```

录制文件包含：

- `joint_pos`
- `joint_vel`
- `mimic_obs`
- `action`
- `target_dof_pos`
- `torque`
- `timestamp`

## 渲染

```bash
MUJOCO_GL=egl python scripts/render_sim.py \
    --bvh data/lafan1/dance1_subject2.bvh \
    --policy policy.onnx
MUJOCO_GL=egl python scripts/render_sim.py \
    --bvh data/hc_mocap/wander.bvh \
    --format hc_mocap \
    --policy policy.onnx
```

`render_sim.py` 的三路输出现在都走 MuJoCo 渲染链路；第一个 pass 是 `mocap` 输入骨架。旧的 `bvh` 可视化命名已移除。

## 继续阅读

- 配置说明：[`configuration.md`](configuration.md)
- 数据准备：[`dataset.md`](dataset.md)
- 训练与导出：[`training.md`](training.md)
- 真机部署：[`sim2real.md`](sim2real.md)
