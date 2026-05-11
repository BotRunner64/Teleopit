---
sidebar_position: 1
---

# 配置概览

Teleopit 使用 [Hydra](https://hydra.cc/) 组合配置。大多数运行入口都会从一个顶层 YAML 开始，再通过命令行 override 修改局部字段。

运行时装配统一放在 `teleopit/runtime/`。脚本、`TeleopPipeline` 和 sim2real 状态机都会复用同一套路径解析、默认值传播和维度校验逻辑。

## 顶层配置

| 配置文件 | 用途 |
|---------|------|
| `teleopit/configs/default.yaml` | 离线 sim2sim（BVH 回放） |
| `teleopit/configs/pico4_sim.yaml` | Pico 4 VR sim2sim |
| `teleopit/configs/sim2real.yaml` | BVH sim2real（Unitree G1 真机） |
| `teleopit/configs/pico4_sim2real.yaml` | Pico 4 VR sim2real（Unitree G1 真机） |

它们会组合以下子配置：

- `teleopit/configs/robot/g1.yaml`
- `teleopit/configs/controller/rl_policy.yaml`
- `teleopit/configs/input/bvh.yaml` — 离线 BVH 输入
- `teleopit/configs/input/pico4.yaml` — 通过 Teleopit host 上的 pico-bridge receiver 接入 Pico 4

## Override 示例

### 基本 Sim2Sim

```bash
python scripts/run/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    policy_hz=50 \
    pd_hz=200
```

### 切换 Viewer

```bash
python scripts/run/run_sim.py controller.policy_path=policy.onnx viewers=all
python scripts/run/run_sim.py controller.policy_path=policy.onnx viewers=none
python scripts/run/run_sim.py controller.policy_path=policy.onnx 'viewers=[retarget,sim2sim]'
```

### 开启键盘回放

```bash
python scripts/run/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    playback.keyboard.enabled=true
```

### 修改网络接口（sim2real）

PC 通过网线连接 G1 控制时，先在 PC 上运行 `ifconfig`，将此字段设置为连接机器人的网线接口名，例如 `enp130s0`。在机器人 onboard 计算机上运行时，默认的 `eth0` 通常是正确值。

```bash
python scripts/run/run_sim2real.py \
    controller.policy_path=policy.onnx \
    real_robot.network_interface=enp130s0
```

## 设计原则：Fail-Fast

Teleopit 不会静默修补配置错误：

- policy 维度不对 → 直接报错
- 观测定义不匹配 → 直接报错
- 必需路径缺失 → 直接报错
- 使用已废弃的配置键 `viewer` → 直接报错
- 不会自动 pad/trim 观测

当你遇到配置错误时，应该查找**哪两个组件的定义不一致**。

完整字段参考请查看 [配置参考](config-reference)。
