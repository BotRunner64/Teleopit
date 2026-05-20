---
sidebar_position: 2
---

# Pico 4 VR 仿真遥操作

使用本教程在接入真实 Unitree G1 之前，先在 MuJoCo 中验证 Pico 4 / Pico 4 Ultra
全身追踪。

```text
Pico 头显 -> pico-bridge receiver -> retarget -> RL policy -> MuJoCo G1
```

此流程跑通后，再继续阅读 [Pico Sim2Real](pico-sim2real)。

## 支持设备

- Pico 4
- Pico 4 Ultra

## 1. 设置头显

1. 从 [pico-bridge Releases](https://github.com/BotRunner64/pico-bridge/releases) 下载头显 APK。
2. 使用 adb 安装：
   ```bash
   adb install pico-bridge.apk
   ```
3. 启动 pico-bridge 头显 client。
4. 开启全身追踪。
5. 确保头显和 Teleopit host 在同一网络。

## 2. 安装 Pico Host Extra

在运行 Teleopit 的机器上执行：

```bash
pip install -e '.[pico4]'
```

验证 receiver 包：

```bash
python -c "from pico_bridge import PicoBridge; print('OK')"
```

Teleopit 会通过 `Pico4InputProvider` 在进程内启动 `pico_bridge.PicoBridge`。
后续 wired 和 onboard sim2real 部署也使用同一条 Pico 输入路径。

## 3. 下载资源

```bash
pip install modelscope
python scripts/setup/download_assets.py --only gmr ckpt bvh
```

## 4. 运行 Pico Sim2Sim

```bash
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=track.onnx
```

仿真从 `STANDING` 开始。等待 Pico 追踪激活后，再进入 `MOCAP`。

| 键盘 | 动作 |
|------|------|
| `Y` | 进入 `MOCAP` |
| `A` | 暂停 / 恢复实时动捕 |
| `X` | 返回 `STANDING` |
| `Q` | 退出 |

`pico4_sim.yaml` 默认使用 `viewers=all`，会打开 mocap、retarget 和 sim2sim
三个 viewer。需要更少窗口时，可使用 `viewers=sim2sim` 或 `viewers=none`。

## 暂停 / 恢复

Pico 暂停/恢复会冻结 mocap session；它不是切回 `STANDING`。

- 按键盘 `A` 或 Pico/controller 暂停键，冻结当前参考姿态。
- 再按一次会重建实时参考路径，重新居中 yaw 和地面平面位置，然后从当前实时追踪流继续。

默认 Pico 暂停键是 `A`。支持的覆盖值包括 `B`、`X`、`Y`、`left_axis_click`、
`right_axis_click`、`left_menu_button` 和 `right_menu_button`。

## 可选头显视频预览

pico-bridge 0.2.0 可以在头显中显示 host 侧视频流。在仿真中，Teleopit 可以推送
MuJoCo `d435i_rgb` 相机：

```bash
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=track.onnx \
    input.video.enabled=true
```

使用 `input.video.source=test-pattern` 可以做 receiver 侧视频 sanity check。如果视频启动失败，
Teleopit 会记录错误、关闭视频，并继续运行追踪和控制。设置
`input.video.fail_on_error=true` 可改为启动失败。

## 常用参数

```bash
# 等待第一帧 Pico body 数据的超时时间
input.pico4_timeout=30

# 覆盖 discovery 广播给头显的 IP
input.bridge_advertise_ip=192.168.1.20

# 关闭 discovery 并显式绑定
input.bridge_discovery=false input.bridge_host=0.0.0.0 input.bridge_port=63901

# 更换 Pico 暂停键
input.pause_button=right_axis_click

# 关闭键盘模式控制
keyboard.enabled=false

# 修改策略频率
policy_hz=30

# 开启头显视频预览
input.video.enabled=true
```

## 故障排查

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| `ImportError: pico_bridge` | 未安装 Pico extra | 执行 `pip install -e '.[pico4]'` |
| 启动提示 pico-bridge 太旧 | 已安装 receiver 不支持视频参数 | 重新安装 Pico extra，确保使用 pico-bridge 0.2.0 |
| `TimeoutError: No Pico4 body data` | 头显未连接或 body tracking 未激活 | 检查头显 app、网络和 `input.pico4_timeout` |
| discovery 找不到 host | 广播 IP 不对或 UDP 被阻断 | 设置 `input.bridge_advertise_ip=<host-ip>`，确认 UDP 端口 `63901` 可达 |
| 仿真机器人不跟随 | 循环仍在 `STANDING` | 追踪准备好后按 `Y` |
| Pico 视频黑屏或被关闭 | 视频源失败或相机不可访问 | 检查 `input.video.source` 和日志 |
