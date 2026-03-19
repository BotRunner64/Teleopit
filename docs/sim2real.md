# Sim2Real 部署指南

本文档介绍如何使用 Teleopit 通过 Unitree SDK2 控制实物 G1。当前真机控制统一使用 `standing/mocap` 双模式状态机。

当前真机路径只支持：

- `Tracking-Flat-G1-VelCmdHistory` 训练得到的策略
- 166D VelCmdHistory 双输入 ONNX（`obs` + `obs_history`）

> 如果你还在熟悉离线/在线推理主路径，先看 [`inference.md`](inference.md)。

## 概述

Sim2Real 复用现有的输入、retarget、观测构建和策略推理管线，只把执行后端从 MuJoCo 换成 Unitree SDK2 DDS。

支持两类实时输入：

| 输入源 | 入口脚本 | 说明 |
|------|--------|------|
| UDP BVH (`hc_mocap`) | `python scripts/run_sim2real.py ...` | 默认真机输入路径 |
| Pico4 | `python scripts/run_sim2real.py --config-name pico4_sim2real ...` | 通过 `xrobotoolkit_sdk` 接收实时 body tracking |

## 控制模式

| 模式 | 数据流 | 适用场景 |
|------|--------|----------|
| `STANDING` | 默认站姿参考 → RL policy → SDK 关节控制 | 起步、恢复、等待进入动捕 |
| `MOCAP` | UDP BVH / Pico4 → retarget → RL policy → SDK 关节控制 | 全身遥操作 |
| `DAMPING` | 退出 debug，发送阻尼命令 | 急停/恢复 |

状态机：

```text
                     ┌─────────────────────────────┐
                     │      L1+R1 急停 (任意状态)    │
                     ▼                             │
  [IDLE] ──Start──▶ [STANDING] ──Y──▶ [MOCAP] ──X──▶ [STANDING]
                           ▲                       │
                           └──────────Y────────────┘
    ▲                                                  │
    └──────────────────Start───────────────────────────┘
                           [DAMPING]
```

## 前置要求

硬件：

- Unitree G1（29 DOF）
- Unitree 无线遥控器
- 机器人与控制 PC 的网络连接
- `hc_mocap` UDP 数据源或 Pico4 上游服务

软件：

- Python 3.10+
- `pip install -e '.[sim2real]'`
- Unitree SDK2 Python submodule
- 如果走 Pico4，还需手动安装 `xrobotoolkit_sdk`

初始化 SDK submodule：

```bash
git submodule update --init --recursive
```

## 快速开始

UDP BVH 真机遥操作：

```bash
# 终端 1
python scripts/run_sim2real.py           controller.policy_path=policy.onnx           real_robot.network_interface=eth0

# 终端 2
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop --port 1118
```

Pico4 真机遥操作：

```bash
python scripts/run_sim2real.py           --config-name pico4_sim2real           controller.policy_path=policy.onnx           real_robot.network_interface=eth0
```

推荐进入流程：

1. 启动控制脚本。
2. 按 `Start` 进入 `STANDING`。
3. 确认实时输入正常。
4. 按 `Y` 切到 `MOCAP`。
5. 按 `X` 回到 `STANDING`；按 `L1+R1` 进入 `DAMPING`。

## 常用参数

```bash
python scripts/run_sim2real.py controller.policy_path=policy.onnx policy_hz=30
python scripts/run_sim2real.py controller.policy_path=policy.onnx input.udp_port=1119
python scripts/run_sim2real.py --config-name pico4_sim2real controller.policy_path=policy.onnx input.pico4_timeout=30
```

## 关键约束

- 真机路径只支持 166D VelCmdHistory 双输入 ONNX。
- `RLPolicyController` 会维护 history buffer，并在 reset 时清空。
- 真机没有 state-estimation fallback；不接受其他观测定义。
- `Pico4InputProvider` 的输入空间变换必须以实际 retarget / sim2sim 结果验证。
- 正常退出和急停都应落到 damping 行为。
