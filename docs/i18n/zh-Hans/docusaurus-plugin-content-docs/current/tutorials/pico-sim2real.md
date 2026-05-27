---
sidebar_position: 4
---

# Pico 4 VR 真机遥操作

在 [Pico Sim2Sim](pico-sim2sim) 跑通后，使用本教程把同一条实时 Pico 输入路径部署到
真实 Unitree G1。

```text
Pico 头显 -> Teleopit host -> retarget -> RL policy -> g1_bridge_sdk -> G1
```

有两种部署方式：

| 部署方式 | Teleopit 运行位置 | 主要区别 |
|----------|-------------------|----------|
| Wired PC-to-G1 | 外部工作站或笔记本 | 将 `real_robot.network_interface` 设置为 PC 上连接 G1 的以太网接口 |
| Onboard | G1 onboard 计算机 | 在 onboard 计算机安装 Teleopit；通常使用 `eth0` |

两种方式都使用 `Pico4InputProvider` 和进程内 pico-bridge receiver。不存在单独的
onboard Pico 输入模式。

Teleopit 面向 pico-bridge 0.2.1 及其 `pico_native` tracking 语义。

## 1. 安装运行时依赖

在运行 Teleopit 的机器上安装 Pico 和 sim2real 依赖：

```bash
pip install -e '.[pico4]'
git submodule update --init --recursive
bash scripts/setup/setup_g1_bridge.sh
```

验证 Pico receiver 导入：

```bash
python -c "from pico_bridge import PicoBridge; print('OK')"
```

## 2. 选择网络接口

`real_robot.network_interface` 是用于 Unitree DDS 通信的 Linux 网卡接口。

对于 wired PC-to-G1 部署：

1. 用网线连接 PC 和 G1。
2. 在 PC 上运行 `ifconfig`。
3. 使用连接到机器人的以太网接口，例如 `enp130s0`。
4. 确保 Pico 头显所在网络可以访问运行 Teleopit 的 PC。

对于 onboard 部署：

1. 在机器人 onboard 计算机上运行 Teleopit。
2. 确保 Pico 头显所在网络可以访问 onboard 计算机。
3. 除非机器人网络不同，否则使用 `real_robot.network_interface=eth0`。
4. 如果 Pico discovery 广播了错误地址，设置 `input.bridge_advertise_ip=<host-ip>`。

### Arm Onboard 的 RealSense 配置

pico-bridge PC receiver 在所需 Python 依赖可用时支持 Arm 机器。对于需要 RealSense
预览的 Arm onboard 计算机，应在当前 Conda 环境中从 conda-forge 安装 `pyrealsense2`，
不要依赖 pip 包：

```bash
pip uninstall pyrealsense2
conda install -c conda-forge pyrealsense2
```

这只影响可选的 RealSense 预览路径（`input.video.enabled=true`）。Pico 追踪和机器人控制
本身不需要 RealSense。

## 3. 运行控制器

Wired PC 示例：

```bash
python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    real_robot.network_interface=enp130s0
```

Onboard 示例：

```bash
python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    real_robot.network_interface=eth0
```

## 操作流程

始终把 Unitree 遥控器拿在手里。`L1+R1` 是进入 `DAMPING` 的急停路径。

| 控制 | 动作 |
|------|------|
| Unitree remote `Start` | 进入 `STANDING` |
| Unitree remote `Y` | 进入 `MOCAP` |
| Pico/controller `A` | 暂停 / 恢复实时动捕 |
| Unitree remote `X` | 返回 `STANDING` |
| Unitree remote `L1+R1` | 急停（`DAMPING`） |

只在 Pico 追踪稳定后进入 `MOCAP`。Teleopit 会在切换前验证连续动捕帧；验证失败时，
机器人会保持在 `STANDING`。

## 运行时行为

Pico sim2real 使用共享的实时参考时间线：

```text
Pico body frames -> retarget -> reference buffer -> observation -> policy -> G1 joints
```

进入 `STANDING` 时，Teleopit 会释放当前 Unitree 模式，进入 debug/low-level 控制，
短暂锁住当前关节，重置 policy 状态，并在不改变 policy target 的情况下执行 Kp ramp。

进入 `MOCAP` 时，Teleopit 会重置 policy/reference 状态，并通过实时参考时间线开始跟踪
实时 mocap 命令。

## 暂停 / 恢复

Pico 暂停/恢复是 mocap-session control event。

- `ACTIVE`：暂停键冻结当前参考姿态。
- `PAUSED`：再次按下会清空 policy/reference 状态，预热实时 buffer，重新居中 yaw/XY 对齐，
  并从实时 mocap 恢复。

:::warning
恢复时请保持静止，并尽量接近暂停时的姿态。这样可以减少实时追踪恢复时的参考突变。
:::

## 可选 LinkerHand L6 控制

Pico sim2real 可以用 Pico 手柄控制 LinkerHand L6。按住同侧 grip 作为 deadman，
同侧 trigger 控制对应手闭合。手控只在 `MOCAP` 中生效；在 `STANDING`、`DAMPING`、
mocap 暂停、帧超时和退出时都会发送张开姿态。

如果主 Pico profile 没有包含手控支持，先安装 dexhand extra：

```bash
pip install -e '.[dexhand]'
```

测试或运行手控前，先开启 CAN 接口：

```bash
sudo /usr/sbin/ip link set can0 up type can bitrate 1000000
sudo /usr/sbin/ip link set can1 up type can bitrate 1000000
```

启用完整 sim2real 前，先用独立开合测试验证灵巧手连接：

```bash
python scripts/dev/test_linkerhand_l6.py \
    --hand-type both \
    --left-can can0 \
    --right-can can1
```

然后在 Pico sim2real 中启用 L6 控制：

```bash
dexterous_hand.enabled=true
dexterous_hand.left_can=can0
dexterous_hand.right_can=can1
```

## 可选 RealSense 预览

将 G1 RealSense 彩色相机推送回 Pico 头显：

```bash
python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    real_robot.network_interface=enp130s0 \
    input.video.enabled=true \
    input.video.device=<optional-realsense-serial>
```

如果视频失败，控制会继续运行，除非设置了 `input.video.fail_on_error=true`。

## 常用参数

```bash
# G1 DDS 网卡接口
real_robot.network_interface=enp130s0

# Pico 超时时间
input.pico4_timeout=30

# 覆盖 Pico discovery 广播 IP
input.bridge_advertise_ip=192.168.1.20

# 进入 MOCAP 前要求的连续有效动捕帧数
mocap_switch.check_frames=10

# 更换 Pico 暂停键
input.pause_button=right_axis_click

# 开启 LinkerHand L6 控制
dexterous_hand.enabled=true

# 开启头显视频预览
input.video.enabled=true
```

## 故障排查

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| 没有收到 LowState | 网卡错误或 G1 网络未连接 | 检查网线和 `real_robot.network_interface` |
| `TimeoutError: No Pico4 body data` | 头显未连接或追踪未激活 | 检查头显 app、网络和 `input.pico4_timeout` |
| 无法进入 debug mode | Unitree mode 释放失败 | 停止其他机器人模式后再次按 `Start` |
| 机器人进入 `STANDING` 但不进入 `MOCAP` | 动捕验证失败 | 保持追踪稳定，查看 `mocap_switch.check_frames` 日志 |
| Pico 暂停没有返回 `STANDING` | 这是预期行为 | Pico 暂停只冻结 mocap；按遥控器 `X` 返回 `STANDING` |
| LinkerHand 不动 | 不在 `MOCAP`、deadman grip 未按住、SDK 未安装，或 CAN 通道错误 | 进入 `MOCAP`，按住同侧 grip，运行 `scripts/dev/test_linkerhand_l6.py`，并检查 `dexterous_hand.left_can` / `right_can` |
| 视频预览不可用 | RealSense 或视频源失败 | 检查相机权限、`input.video.source` 和日志 |
