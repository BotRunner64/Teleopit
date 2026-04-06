---
sidebar_position: 4
---

# 机载 Onboard Sim2Real

在 G1 机载 NX 计算机上直接运行 sim2real 控制回路，Pico 4 追踪数据通过 ZMQ 从外部 PC 传入。

## 网络拓扑

通过路由器连接两台 PC：

```text
 [路由器] --- WLAN/WiFi --- [PC2: 用户 PC]
    |
    +--- WLAN/WiFi --- [PC1: G1 NX 机载计算机]
```

- **PC1**（G1 NX）：通过 WiFi 连接到路由器
- **PC2**（用户 PC）：通过 WiFi 连接到同一路由器

## 配置固定 IP

1. 在路由器管理页面分配固定 IP，例如：
   - PC1（NX）：`192.168.1.101`
   - PC2（用户 PC）：`192.168.1.102`
2. 也可在各设备上手动配置静态 IP

## 验证连通性

```bash
# 从 PC2 ping PC1
ping 192.168.1.101

# 从 PC2 SSH 到 PC1
ssh user@192.168.1.101
```

## 安装环境（NX 端）

```bash
# SSH 到 NX 后，在 Teleopit 仓库目录下执行
bash scripts/setup/setup_onboard.sh
```

该脚本会安装系统依赖、编译 `g1_bridge_sdk`，并安装 `teleopit[onboard]`。

## 运行机载 Sim2Real

**在 PC2（用户 PC）上** — 启动 Pico 4 ZMQ 数据转发：

```bash
python scripts/dev/zmq_pico4_publisher.py --bind 0.0.0.0 --port 5555
```

该脚本读取 Pico 4 全身追踪数据并通过 ZMQ 发布，保持运行即可。

**在 PC1（NX）上** — 启动机载控制回路：

```bash
python scripts/run/run_onboard_sim2real.py \
    controller.policy_path=track.onnx \
    real_robot.network_interface=wlan0 \
    input.zmq_host=192.168.1.102
```

其中 `input.zmq_host` 为 PC2 的 IP 地址（即运行 ZMQ 数据转发的机器）。

## 操作说明

控制流程和遥控器按键映射与 [Sim2Real 部署](sim2real#遥控器按键映射) 完全一致。唯一区别在于控制回路运行在 NX 机载计算机上，而非外部 PC。
