---
sidebar_position: 2
---

# Pico 4 VR 全身追踪遥操作

使用 Pico 4 / Pico 4 Ultra 头显的全身追踪功能，实现对机器人的实时遥操作。

## 支持设备

- Pico 4
- Pico 4 Ultra

## 系统架构

```text
Pico 头显（pico-bridge client，全身追踪）
    --WiFi--> PC（pico-bridge PC receiver）
                --> Teleopit (retarget -> RL policy -> MuJoCo / G1)
```

## 第一步：VR 头显设置

1. 从 [pico-bridge Releases](https://github.com/BotRunner64/pico-bridge/releases) 下载头显端 APK
2. 通过 adb 安装：
   ```bash
   adb install pico-bridge.apk
   ```
3. 启动 pico-bridge 头显端 client，并开启**全身追踪**模式
4. 确保头显与控制 PC 处于**同一网络**

## 第二步：PC 端环境配置

### 前置要求

- Ubuntu 22.04
- Python 3.10+
- 安装带 Pico extra 的 Teleopit：`pip install -e '.[pico4]'`

### 安装 pico-bridge PC Receiver

```bash
pip install -e '.[pico4]'
```

验证安装：
```bash
python -c "from pico_bridge import PicoBridge; print('OK')"
```

Teleopit 会在 `Pico4InputProvider` 中启动 PC receiver。

## 第三步：仿真验证（Pico sim2sim）

如果尚未下载模型资源，请先执行（详见[下载资源](../getting-started/download-assets)）：

```bash
pip install modelscope
python scripts/setup/download_assets.py --only gmr ckpt bvh
```

在部署真机前，先在 MuJoCo 仿真中验证 VR 追踪数据和策略推理：

```bash
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=track.onnx
```

正常运行后，虚拟机器人会跟随你的 VR 动作。如果机器人没有响应，请检查：
- 头显上的 pico-bridge client 已连接到 PC receiver
- `input.bridge_host`、`input.bridge_port` 以及可选的 `input.bridge_advertise_ip` 与当前网络匹配
- 两台设备处于同一网络

### 键盘模式流程

在 `teleopit/configs/pico4_sim.yaml` 中，实时键盘模式默认开启：

- 按 **Y** 进入 `MOCAP`
- 按 **A** 暂停/恢复实时动捕
- 按 **X** 返回 `STANDING`
- 按 **Q** 退出仿真循环

循环会直接进入 `STANDING`，等追踪准备好后按 **Y** 即可进入 `MOCAP`。

### 暂停与恢复

- 按键盘 **A** 或 Pico 手柄 **A** 键冻结追踪
- 再次按 **A** 键恢复追踪。Teleopit 会先重新居中航向和地面平面位置，然后继续跟随。

## 第四步：真机部署（Pico sim2real）

仿真验证通过后，连接 Unitree G1 真机：

```bash
pip install -e '.[sim2real]'
git submodule update --init --recursive

python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    real_robot.network_interface=eth0
```

### 操作流程

1. 启动脚本
2. 按遥控器 **Start** → 进入 `STANDING`（机器人站立）
3. 确认 Pico 追踪数据正常到达（查看终端日志）
4. 按 **Y** → 进入 `MOCAP`（开始遥操作）
5. 按手柄 **A** 暂停/恢复追踪
6. 按 **X** → 返回 `STANDING`
7. **L1+R1** → 急停（`DAMPING`）

:::warning
从暂停恢复时，请在新的动捕帧到达期间保持静止，并尽量贴近暂停时的姿态。这样可以减少恢复追踪时的参考突变。
:::

完整状态机文档请参见 [Sim2Real 部署](sim2real)。

## 常用参数

```bash
# 调整 Pico 等待超时时间（默认 60s）
input.pico4_timeout=30

# 调整恢复前采集的新追踪帧数
pause_resume_warmup_steps=2

# 关闭实时键盘模式状态机
keyboard.enabled=false

# 修改策略推理频率
policy_hz=30

# 更换暂停按键
input.pause_button=right_axis_click

# 指定网络接口
real_robot.network_interface=enp3s0
```

## 故障排查

| 现象 | 可能原因 | 解决方法 |
|------|---------|---------|
| `ImportError: pico_bridge` | PC receiver 包未安装 | 执行 `pip install -e '.[pico4]'` |
| `TimeoutError: No Pico4 body data` | 头显未连接或未开启追踪 | 检查 pico-bridge 头显 app 状态和网络连接 |
| 机器人不跟随 VR 动作 | 仍处于 STANDING 模式 | 按遥控器 **Y** 进入 MOCAP |
| 发现广播找不到 PC | 网卡不对或 UDP 被阻断 | 设置 `input.bridge_advertise_ip=<pc-ip>`，确认 UDP 端口 `63901` 可达 |
