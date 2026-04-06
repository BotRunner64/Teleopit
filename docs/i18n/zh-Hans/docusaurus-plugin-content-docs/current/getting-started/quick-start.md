---
sidebar_position: 3
---

# 快速上手

本指南带你在 5 分钟内完成第一次 sim2sim 回放。

## 前置条件

1. [安装 Teleopit](installation)（推理配置）
2. [下载资源](download-assets)（`--only gmr ckpt bvh`）

## 运行离线 Sim2Sim

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

运行后你应该能看到 MuJoCo 查看器窗口，展示机器人跟踪 BVH 动作的过程。

## 键盘控制

在启用 `playback.keyboard.enabled=true` 时可使用以下快捷键：

| 按键 | 功能 |
|------|------|
| `Space` / `P` | 暂停 / 继续 |
| `R` | 从头重播 |
| `Q` | 停止 |

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    playback.keyboard.enabled=true
```

## 查看器模式

控制显示哪些查看器窗口：

```bash
# 显示全部查看器（动捕 + 重定向 + sim2sim）
python scripts/run/run_sim.py controller.policy_path=track.onnx viewers=all

# 无查看器（无头模式）
python scripts/run/run_sim.py controller.policy_path=track.onnx viewers=none

# 指定查看器
python scripts/run/run_sim.py controller.policy_path=track.onnx 'viewers=[retarget,sim2sim]'
```

## 下一步

- [离线 Sim2Sim 教程](../tutorials/offline-sim2sim) - 包含录制和渲染的完整指南
- [Pico 4 VR](../tutorials/pico4-vr) - 实时 VR 遥操作
- [Sim2Real](../tutorials/sim2real) - 部署到 Unitree G1 实物
- [训练](../tutorials/training) - 训练你自己的策略
