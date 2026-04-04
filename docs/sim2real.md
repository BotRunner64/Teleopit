# Sim2Real 部署指南

本文档介绍如何使用 Teleopit 通过 g1_bridge_sdk (C++ DDS bridge) 控制实物 G1。

> **Pico VR 用户**：完整的 Pico 4 / Pico 4 Ultra 部署指南见 **[pico4.md](pico4.md)**。

> 离线推理见 [inference.md](inference.md)，技术规格见 [architecture.md](architecture.md)。

## 输入源

| 输入源 | 配置 | 文档 |
|------|--------|------|
| **Pico 4 / Pico 4 Ultra** | `--config-name pico4_sim2real` | **[Pico VR 部署](pico4.md)** |
| 离线 BVH 动作文件 | 默认配置 | 本文档 |

## 前置要求

硬件：

- Unitree G1（29 DOF）
- Unitree 无线遥控器
- 机器人与控制 PC 的网络连接
- Pico 4 / Pico 4 Ultra 头显，或离线 BVH 动作文件

软件：

- Python 3.10+
- g1_bridge_sdk（C++ DDS bridge）：
  ```bash
  pip install pybind11
  pip install third_party/g1_bridge_sdk
  ```
- `pip install -e '.[sim2real]'`
- Pico 路径额外需要：`bash scripts/setup_pico4.sh`

## 离线动作文件播放

```bash
python scripts/run_sim2real.py \
    controller.policy_path=track.onnx \
    real_robot.network_interface=eth0 \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

默认遥控器映射：

- `Start`：进入 `STANDING`
- `Y`：从 `STANDING` 进入离线播放
- `A`：暂停 / 恢复当前播放
- `B`：从头重播当前动作文件
- `X`：回到 `STANDING`
- `L1+R1`：急停进入 `DAMPING`

## Onboard 部署

在 G1 机载 NX 上运行 sim2real 控制，Pico4 追踪数据通过 ZMQ 从上位机发送。

### 网络拓扑

需要一台路由器通过网线连接 WLAN，或无线桥接 WiFi：

```text
 [路由器] ─── WLAN/WiFi ─── [PC2 用户 PC]
    │
    └─── WLAN/WiFi ─── [PC1 G1 NX 机载电脑]
```

- **PC1** (G1 NX 机载电脑)：无线连接路由器
- **PC2** (用户 PC)：无线连接同一路由器网络

### 配置固定 IP

1. 在路由器管理页面为 PC1 和 PC2 分配固定 IP，例如：
   - PC1 (NX)：`192.168.1.101`
   - PC2 (用户 PC)：`192.168.1.102`
2. 或在各设备上手动配置静态 IP

### 验证连通性

```bash
# 从 PC2 ping PC1
ping 192.168.1.101

# 从 PC2 SSH 连接 PC1
ssh user@192.168.1.101
```

确保 PC1 和 PC2 能相互 ping 通，并且从 PC2 可以通过 SSH 连接 PC1。

### 环境安装（NX 上执行）

```bash
# SSH 到 NX 后，在 Teleopit 仓库目录执行一键安装
bash scripts/setup_onboard.sh
```

此脚本会安装系统依赖、构建 g1_bridge_sdk 并安装 teleopit[onboard]。

### 运行 Onboard Sim2Real

```bash
# PC1 (NX) 上运行
python scripts/run_onboard_sim2real.py \
    controller.policy_path=track.onnx \
    real_robot.network_interface=wlan0 \
    input.zmq_host=192.168.1.102
```

其中 `input.zmq_host` 填写 PC2 的 IP 地址。

## 控制模式

| 模式 | 数据流 | 适用场景 |
|------|--------|----------|
| `STANDING` | 默认站姿 → RL policy → 关节控制 | 起步、恢复、等待 |
| `MOCAP` | Pico / 离线 BVH → retarget → RL policy → 关节控制 | 全身遥操作 / 动作播放 |
| `DAMPING` | 发送阻尼命令 | 急停 |

对于 `input.provider=pico4` 的真机遥操作，`MOCAP` 内部还包含会话子状态：
- `ACTIVE`：正常跟随 live mocap
- `PAUSED`：冻结当前参考姿态，机器人持续平衡但不再跟随人体移动
- `RESUMING`：清空 realtime reference buffer、重建 yaw/pivot 对齐后，平滑接回 live mocap

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
3. 确认输入正常（Pico 追踪已连接 / 离线 BVH 文件已指定）
4. 按 `Y` 切到 `MOCAP`
5. 按 `A` 可暂停/恢复；恢复时尽量保持姿态接近暂停时，若出现扭曲请立即再次暂停；离线播放按 `B` 可从头重播
6. 按 `X` 回到 `STANDING`；`L1+R1` 急停进入 `DAMPING`

## 常用参数

```bash
# 调整控制频率
python scripts/run_sim2real.py controller.policy_path=track.onnx policy_hz=30

# 指定离线 BVH 文件
python scripts/run_sim2real.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh

# Pico 超时时间
python scripts/run_sim2real.py --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    input.pico4_timeout=30

# 修改 Pico 暂停按键
python scripts/run_sim2real.py --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    input.pause_button=right_axis_click

# 调整暂停恢复过渡
python scripts/run_sim2real.py --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    pause_resume_transition_duration=1.5 \
    pause_resume_warmup_steps=3

# 指定网卡
python scripts/run_sim2real.py \
    controller.policy_path=track.onnx \
    real_robot.network_interface=enp3s0
```

## 独立站立测试

`standalone_standing.py` 是一个不依赖 Teleopit 主框架的独立测试脚本，可用于快速验证机器人硬件和 RL policy：

```bash
python scripts/standalone_standing.py \
    --policy track.onnx \
    --network-interface eth0
```

支持 `--dry-run` 模式进行安全的 timing benchmark（不发送电机命令）。
