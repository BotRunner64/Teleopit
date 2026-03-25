# Sim2Real 部署指南

本文档介绍如何使用 Teleopit 通过 Unitree SDK2 控制实物 G1。

> **Pico VR 用户**：完整的 Pico 4 / Pico 4 Ultra 部署指南见 **[pico4.md](pico4.md)**。

> 离线/在线推理见 [inference.md](inference.md)，技术规格见 [architecture.md](architecture.md)。

## 输入源

| 输入源 | 配置 | 文档 |
|------|--------|------|
| **Pico 4 / Pico 4 Ultra** | `--config-name pico4_sim2real` | **[Pico VR 部署](pico4.md)** |
| UDP BVH | 默认配置 | 本文档 |

## UDP BVH 部署

```bash
# 终端 1
python scripts/run_sim2real.py \
    controller.policy_path=track.onnx \
    real_robot.network_interface=eth0

# 终端 2
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop --port 1118
```

## 控制模式

| 模式 | 数据流 | 适用场景 |
|------|--------|----------|
| `STANDING` | 默认站姿 → RL policy → SDK 关节控制 | 起步、恢复、等待 |
| `MOCAP` | Pico / UDP → retarget → RL policy → SDK 关节控制 | 全身遥操作 |
| `DAMPING` | 发送阻尼命令 | 急停 |

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

操作流程：

1. 启动控制脚本
2. 按 `Start` 进入 `STANDING`
3. 确认输入正常（Pico 追踪已连接 / UDP 数据已到达）
4. 按 `Y` 切到 `MOCAP`
5. 按 `X` 回到 `STANDING`；`L1+R1` 急停进入 `DAMPING`

## 前置要求

硬件：

- Unitree G1（29 DOF）
- Unitree 无线遥控器
- 机器人与控制 PC 的网络连接
- Pico 4 / Pico 4 Ultra 头显，或外部动捕 UDP 数据源

软件：

- Python 3.10+、`pip install -e '.[sim2real]'`
- Unitree SDK2 Python submodule：`git submodule update --init --recursive`
- Pico 路径额外需要：`bash scripts/setup_pico4.sh`

## 常用参数

```bash
# 调整控制频率
python scripts/run_sim2real.py controller.policy_path=track.onnx policy_hz=30

# 调整 UDP 端口
python scripts/run_sim2real.py \
    controller.policy_path=track.onnx \
    input.udp_port=1119

# Pico 超时时间
python scripts/run_sim2real.py --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    input.pico4_timeout=30

# 指定网卡
python scripts/run_sim2real.py \
    controller.policy_path=track.onnx \
    real_robot.network_interface=enp3s0
```
