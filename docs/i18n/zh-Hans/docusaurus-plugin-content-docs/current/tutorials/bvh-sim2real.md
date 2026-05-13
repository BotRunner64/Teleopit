---
sidebar_position: 5
---

# Unitree G1 BVH 动作回放

使用本教程在真实 Unitree G1 上回放离线 BVH 动作。此路径不使用 Pico 追踪。

```text
BVH file -> retarget -> RL policy -> g1_bridge_sdk -> G1
```

实时 Pico 控制请使用 [Pico Sim2Real](pico-sim2real)。

## 1. 安装运行时依赖

```bash
pip install -e '.[sim2real]'
git submodule update --init --recursive
bash scripts/setup/setup_g1_bridge.sh
```

## 2. 选择网络接口

对于 wired PC-to-G1 部署，用网线连接 PC 和机器人，运行 `ifconfig`，并使用连接到 G1 的接口：

```bash
real_robot.network_interface=enp130s0
```

对于 onboard 部署，通常使用 `eth0`。

## 3. 运行 BVH 回放

```bash
python scripts/run/run_sim2real.py \
    controller.policy_path=track.onnx \
    real_robot.network_interface=enp130s0 \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

默认 `sim2real.yaml` 使用 `input.provider=bvh` 和 `playback.pause_on_end=true`，
因此动作结束时会保持最后一帧姿态。

## 遥控器控制

| 按键 | 动作 |
|------|------|
| `Start` | 进入 `STANDING` |
| `Y` | 进入回放 / `MOCAP` |
| `A` | 暂停 / 恢复回放 |
| `B` | 从第 0 帧重播 |
| `X` | 返回 `STANDING` |
| `L1+R1` | 急停（`DAMPING`） |

## 回放行为

- `Y` 从当前回放位置启动 BVH 动作。
- `A` 在当前参考姿态暂停，并从同一时间线恢复。
- `B` 将回放重置到第 0 帧，并重启 policy/reference 状态。
- 如果 `playback.pause_on_end=true`，最后一帧会保持到按 `B` 重播或按 `X` 返回
  `STANDING`。

## 常用参数

```bash
# BVH 文件
input.bvh_file=data/sample_bvh/aiming1_subject1.bvh

# G1 DDS 网卡接口
real_robot.network_interface=enp130s0

# 在 BVH 最后一帧暂停
playback.pause_on_end=true

# 从 standing/当前机器人状态平滑进入回放
transition_duration=2.0

# 离线回放恢复混合时长
pause_resume_transition_duration=1.0

# 控制循环频率
policy_hz=50
```

## 故障排查

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| 没有收到 LowState | 网络接口错误 | 检查网线和 `real_robot.network_interface` |
| 机器人进入 `STANDING` 但不回放 | BVH 验证失败 | 检查 `input.bvh_file` 和 retarget 日志 |
| 回放结束后机器人保持姿态 | `playback.pause_on_end=true` | 按 `B` 重播或按 `X` 返回 `STANDING` |
| `B` 没有效果 | 不在离线 BVH MOCAP 模式 | 先按 `Y` 进入 `MOCAP` |
