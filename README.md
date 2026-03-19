# Teleopit

Teleopit 是一个轻量、可扩展、自包含的人形机器人全身遥操作框架。

当前仓库已经收敛到单一正式主线：

- 训练侧只保留 `Tracking-Flat-G1-VelCmdHistory`
- 推理侧只保留 **166D VelCmdHistory 双输入 ONNX**（`obs` + `obs_history`）
- adaptive sampling、旧单帧/非 VelCmdHistory 推理路径和其他历史 task 已清理移除

## Architecture

```text
InputProvider (BVH file / UDP realtime / Pico4)
    -> Retargeter (GMR)
    -> ObservationBuilder (166D VelCmdHistory)
    -> Controller (dual-input ONNX RL policy)
    -> Robot (MuJoCo sim or Unitree G1)
```

离线/在线推理由 `teleopit/runtime/` 和 `teleopit/pipeline.py` 装配，真机状态机保留在 `teleopit/sim2real/controller.py`。训练闭环由 `train_mimic/` 提供，导出的唯一受支持策略格式是 VelCmdHistory TemporalCNN ONNX。

## Install

项目要求 Python `3.10+`。

```bash
pip install -e .
pip install -e '.[train]'
pip install -e '.[sim2real]'
pip install -e '.[train,sim2real]'
```

运行测试：

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

## Quick Start

离线 sim2sim：

```bash
python scripts/run_sim.py           controller.policy_path=policy.onnx           input.bvh_file=data/lafan1/dance1_subject2.bvh
```

UDP 实时 online sim2sim：

```bash
# Terminal 1
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx

# Terminal 2
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop
```

Pico4 sim2sim：

```bash
python scripts/run_sim.py --config-name pico4_sim controller.policy_path=policy.onnx
```

G1 sim2real：

```bash
python scripts/run_sim2real.py controller.policy_path=policy.onnx
```

Pico4 sim2real：

```bash
python scripts/run_sim2real.py --config-name pico4_sim2real controller.policy_path=policy.onnx
```

训练：

```bash
python train_mimic/scripts/train.py           --motion_file data/datasets/twist2_full/train.npz
```

导出 ONNX：

```bash
python train_mimic/scripts/save_onnx.py           --checkpoint logs/rsl_rl/g1_tracking_velcmd_history/<run>/model_30000.pt           --output policy.onnx           --history_length 10
```

## Supported Surface

- 唯一训练 task：`Tracking-Flat-G1-VelCmdHistory`
- 唯一推理观测：166D VelCmdHistory
- 唯一 ONNX 签名：`obs` + `obs_history`
- 训练采样模式：`uniform`
- playback / benchmark 采样模式：`start`

## Constraints

- `controller.policy_path` 必须显式提供，且文件必须存在。
- 运行时只接受 166D 双输入 VelCmdHistory ONNX；旧 TWIST2、单输入或其他维度 ONNX 会直接报错。
- 观测定义与 ONNX 输入不匹配时 fail fast；不会自动 pad/trim。
- 离线 BVH 运行必须显式传 `input.bvh_file=...`。
- viewer 只接受 `viewers` 配置键；旧 `viewer` alias 已移除。
- Pico4 依赖 `xrobotoolkit_sdk`，不会随 `pip install -e .` 自动安装。
- sim2sim 默认应使用 `g1_mjlab.xml`；`g1_mocap_29dof.xml` 仅用于运动学可视化。

## Docs

- [`docs/getting-started.md`](docs/getting-started.md)：最短上手路径
- [`docs/inference.md`](docs/inference.md)：离线/在线推理、viewer、录制、渲染
- [`docs/training.md`](docs/training.md)：单 task 训练、播放、benchmark、导出 ONNX
- [`docs/sim2real.md`](docs/sim2real.md)：Unitree G1 实机部署与状态机
- [`docs/dataset.md`](docs/dataset.md)：数据集构建主线
- [`docs/configuration.md`](docs/configuration.md)：Hydra 配置入口
- [`docs/architecture.md`](docs/architecture.md)：系统边界和装配关系
- [`docs/training_troubleshooting.md`](docs/training_troubleshooting.md)：训练问题排查
