---
sidebar_position: 99
---

# Sim2Real 概览

Sim2Real 控制使用 `g1_bridge_sdk`，也就是 Unitree G1 的 C++ DDS bridge。使用本页选择合适的
真机教程。

| 目标 | 教程 |
|------|------|
| 不接 mocap，先验证 bridge、网络和 policy 站立 | [独立站立测试](standalone-standing) |
| 使用 Pico 全身追踪控制真实 G1 | [Pico Sim2Real](pico-sim2real) |
| 在真实 G1 上回放离线 BVH 动作 | [BVH Sim2Real Playback](bvh-sim2real) |

## 共同概念

`real_robot.network_interface` 必须指向用于 Unitree DDS 通信的 Linux 网络接口。

| 部署方式 | Teleopit 运行位置 | 常见接口 |
|----------|-------------------|----------|
| Wired PC-to-G1 | 通过网线连接 G1 的外部 PC | `enp...`，例如 `enp130s0` |
| Onboard | G1 onboard 计算机 | `eth0` |

Unitree 遥控器控制真机状态机：

| 按键 | 动作 |
|------|------|
| `Start` | 从 `IDLE` 或 `DAMPING` 进入 `STANDING` |
| `Y` | 从 `STANDING` 进入 `MOCAP` |
| `X` | 从 `MOCAP` 返回 `STANDING` |
| `L1+R1` | 急停进入 `DAMPING` |

始终先在仿真中验证 policy，手持 Unitree 遥控器，并且只在输入数据稳定后进入 `MOCAP`。
