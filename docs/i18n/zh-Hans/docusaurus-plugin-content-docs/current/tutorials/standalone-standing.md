---
sidebar_position: 3
---

# 独立站立测试

在接入完整 sim2real 控制之前，如果正在调试新机器人、网络配置或 policy，先运行此测试。
它不使用 Pico、BVH 回放、retargeting，也不走完整 Teleopit mocap pipeline，只验证 G1 bridge
和 RL standing 路径。

```text
G1 LowState -> standing observation -> RL policy -> G1 LowCmd targets
```

## 何时使用

适合在以下场景使用 standalone 测试：

- 正在设置新的 wired 或 onboard G1 部署
- `run_sim2real.py` 涉及的组件太多，不适合一次性排查
- 需要先验证 `g1_bridge_sdk`、网络接口选择和 policy 推理，再启用 mocap

## 安装运行时依赖

```bash
pip install -e '.[sim2real]'
git submodule update --init --recursive
bash scripts/setup/setup_g1_bridge.sh
```

## Dry Run

先使用 `--dry-run` 做不发送电机命令的时序检查：

```bash
python scripts/run/standalone_standing.py \
    --policy track.onnx \
    --network-interface enp130s0 \
    --dry-run
```

## 真机站立测试

确认网络接口后，运行 standing controller：

```bash
python scripts/run/standalone_standing.py \
    --policy track.onnx \
    --network-interface enp130s0
```

对于 onboard 部署，接口通常是 `eth0`：

```bash
python scripts/run/standalone_standing.py \
    --policy track.onnx \
    --network-interface eth0
```

standalone standing 使用与 sim2real 相同的 Kp ramp 语义：锁住当前关节后立即发送
policy target，同时在 2 秒内把 Kp 从 10% 逐步升到配置的增益。可以这样调整启动行为：

```bash
python scripts/run/standalone_standing.py \
    --policy track.onnx \
    --network-interface eth0 \
    --kp-ramp-duration 2.0 \
    --kp-ramp-floor-ratio 0.1
```

## 它会检查什么

- `g1_bridge_sdk` 能正确导入。
- 能从机器人收到 LowState。
- dual-input ONNX policy 能运行 standing observation 路径。
- 能通过 C++ bridge 发布 low-level position targets。

## 下一步

standalone 站立测试跑通后：

- 使用 [Pico Sim2Real](pico-sim2real) 进行实时 VR 遥操作
- 使用 [BVH Sim2Real Playback](bvh-sim2real) 进行离线动作回放

## 故障排查

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| 没有收到 LowState | 接口错误或 G1 网络未连接 | 检查网线和 `--network-interface` |
| `g1_bridge_sdk` 导入错误 | bridge 未构建或未安装 sim2real extra | 重新执行安装和 bridge setup 命令 |
| policy shape 错误 | ONNX 导出不匹配 | 使用 `save_onnx.py` 导出的 dual-input TemporalCNN policy |
