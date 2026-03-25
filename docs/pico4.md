# Pico VR 全身追踪部署

本文档介绍如何使用 Pico 4 / Pico 4 Ultra 的全身追踪能力进行实时遥操作，覆盖 VR 端设置、PC 端环境、仿真验证和真机部署完整流程。

## 支持设备

- **Pico 4**
- **Pico 4 Ultra**

## 总体架构

```text
Pico 头显 (XRoboToolkit App, 全身追踪)
    ──WiFi──▶ PC (XRoboToolkit PC Service + xrobotoolkit_sdk)
                ──▶ Teleopit (retarget → RL policy → MuJoCo / G1 真机)
```

---

## 第一步：VR 头显端设置

1. 从 [XRoboToolkit-Unity-Client Releases](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases) 下载最新 APK
2. 通过 adb 安装到 Pico 头显：
   ```bash
   adb install XRoboToolkit-Unity-Client.apk
   ```
3. 在头显中启动 XRoboToolkit 应用，开启 **全身追踪（Full Body Tracking）** 模式
4. 确保头显和控制 PC 在 **同一局域网**

## 第二步：PC 端环境配置

### 前置条件

- Ubuntu 22.04
- Python 3.10+，已安装 Teleopit：`pip install -e .`
- pybind11：`conda install -c conda-forge pybind11`
- [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) deb 包：
  ```bash
  wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
  sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
  ```

### 一键安装 SDK

```bash
bash scripts/setup_pico4.sh
```

脚本会自动完成：
- 检测 `libPXREARobotSDK.so` 是否已安装
- 注册动态库到系统链接器路径
- 编译并安装 `xrobotoolkit_sdk` Python 绑定

验证安装：
```bash
python -c "import xrobotoolkit_sdk; print('OK')"
```

## 第三步：仿真验证（Pico sim2sim）

先在 MuJoCo 仿真中确认 VR 追踪数据和策略推理正常：

```bash
python scripts/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=track.onnx
```

此时应能在 viewer 中看到虚拟机器人跟随你的 VR 动作。如果机器人没有响应，检查：
- Pico 头显中 XRoboToolkit 是否显示 "已连接"
- PC 端 PC Service 是否正在运行
- 两台设备是否在同一网络

## 第四步：真机部署（Pico sim2real）

确认仿真效果正常后，连接 Unitree G1 真机：

```bash
pip install -e '.[sim2real]'
git submodule update --init --recursive

python scripts/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    real_robot.network_interface=eth0
```

### 操作流程

1. 启动上述脚本
2. 按遥控器 **Start** → 进入 `STANDING`（机器人站立）
3. 确认 Pico 追踪数据已到达（终端会有日志）
4. 按 **Y** → 进入 `MOCAP`（开始遥操作）
5. 按 **X** → 回到 `STANDING`
6. **L1+R1** → 急停（`DAMPING`）

状态机详情见 [sim2real.md](sim2real.md)。

## 常用参数

```bash
# 调整 Pico 等待超时（默认 60 秒）
python scripts/run_sim.py --config-name pico4_sim \
    controller.policy_path=track.onnx \
    input.pico4_timeout=30

# 调整策略频率
python scripts/run_sim2real.py --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    policy_hz=30

# 指定网卡
python scripts/run_sim2real.py --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    real_robot.network_interface=enp3s0
```

## 故障排查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| `ImportError: xrobotoolkit_sdk` | SDK 未安装 | 运行 `bash scripts/setup_pico4.sh` |
| `TimeoutError: No Pico4 body data` | 头显未连接或追踪未启动 | 检查 XRoboToolkit 应用状态和网络 |
| 机器人不跟随 VR 动作 | 仍在 STANDING 模式 | 按遥控器 Y 进入 MOCAP |
| `libPXREARobotSDK.so not found` | PC Service 未安装 | 安装 deb 包后重新运行 setup 脚本 |
