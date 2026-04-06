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
Pico 头显 (XRoboToolkit App, 全身追踪)
    --WiFi--> PC (XRoboToolkit PC Service + xrobotoolkit_sdk)
                --> Teleopit (retarget -> RL policy -> MuJoCo / G1)
```

## 第一步：VR 头显设置

1. 从 [XRoboToolkit-Unity-Client Releases](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases) 下载最新 APK
2. 通过 adb 安装：
   ```bash
   adb install XRoboToolkit-Unity-Client.apk
   ```
3. 在头显上启动 XRoboToolkit，开启**全身追踪**模式
4. 确保头显与控制 PC 处于**同一网络**

## 第二步：PC 端环境配置

### 前置要求

- Ubuntu 22.04
- Python 3.10+，已安装 Teleopit：`pip install -e .`
- pybind11：`conda install -c conda-forge pybind11`
- [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)：
  ```bash
  wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
  sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
  ```

### 安装 SDK

```bash
bash scripts/setup/setup_pico4.sh
```

该脚本会完成以下操作：
- 检测 `libPXREARobotSDK.so` 是否已安装
- 向系统链接器注册动态库
- 编译并安装 `xrobotoolkit_sdk` Python 绑定

验证安装：
```bash
python -c "import xrobotoolkit_sdk; print('OK')"
```

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
- 头显上 XRoboToolkit 显示 "Connected"
- PC Service 正在运行
- 两台设备处于同一网络

### 暂停与恢复

- 按手柄 **A** 键冻结追踪
- 再次按 **A** 键清除实时参考缓冲区，平滑恢复实时动捕

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
从暂停恢复时，请尽量保持静止并使身体姿态与暂停时的姿态一致。如果出现姿态畸变，请立即再次暂停，调整姿态后重新恢复。
:::

完整状态机文档请参见 [Sim2Real 部署](sim2real)。

## 常用参数

```bash
# 调整 Pico 等待超时时间（默认 60s）
input.pico4_timeout=30

# 调整暂停/恢复过渡时长
pause_resume_transition_duration=1.0

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
| `ImportError: xrobotoolkit_sdk` | SDK 未安装 | 执行 `bash scripts/setup/setup_pico4.sh` |
| `TimeoutError: No Pico4 body data` | 头显未连接或未开启追踪 | 检查 XRoboToolkit 应用状态和网络连接 |
| 机器人不跟随 VR 动作 | 仍处于 STANDING 模式 | 按遥控器 **Y** 进入 MOCAP |
| `libPXREARobotSDK.so not found` | PC Service 未安装 | 安装 deb 包后重新执行安装脚本 |
