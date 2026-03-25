# Teleopit

Teleopit 是一个轻量、可扩展、自包含的人形机器人全身遥操作框架。

训练任务为 `General-Tracking-G1`，对应 `velcmd_history` 166D 观测和 dual-input TemporalCNN ONNX（`obs` + `obs_history`），使用较大的 TemporalCNN actor/critic（1024,512,256,256,128）。

## Architecture

```text
InputProvider (BVH file / UDP realtime / Pico4)
    -> Retargeter (GMR)
    -> ObservationBuilder (166D)
    -> Controller (dual-input TemporalCNN ONNX)
    -> Robot (MuJoCo sim or Unitree G1)
```

离线/在线推理由 `teleopit/runtime/` 和 `teleopit/pipeline.py` 装配，真机状态机保留在 `teleopit/sim2real/controller.py`。训练闭环由 `train_mimic/` 提供。

在线 realtime 路径会先把 retarget 后的 `qpos` 写入短时 reference timeline，再按控制时刻从 timeline 采样 reference window。默认配置使用 `reference_steps=[0]`。

Realtime 配置还会先做短暂 warmup，再按 low/high watermark 维持 reference buffer 水位。推理侧的 `motion_joint_vel`、anchor 线速度和角速度也支持 EMA 平滑，入口配置为 `realtime_buffer_warmup_steps`、`realtime_buffer_low_watermark_steps`、`realtime_buffer_high_watermark_steps`、`reference_velocity_smoothing_alpha`、`reference_anchor_velocity_smoothing_alpha`。

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
python scripts/run_sim.py   controller.policy_path=policy.onnx   input.bvh_file=data/lafan1/dance1_subject2.bvh
```

UDP 实时 sim2sim：

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

训练：

```bash
python train_mimic/scripts/train.py   --motion_file data/datasets/twist2_full/train.npz
```

训练 CLI 约定：

- `--max_iterations` 表示这次调用要追加训练多少轮
- 例如从 `model_12000.pt` resume 且传 `--max_iterations 18000`，训练会继续到 `model_30000.pt`

导出 ONNX：

```bash
python train_mimic/scripts/save_onnx.py   --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt   --output policy.onnx   --history_length 10
```

## Supported Surface

- 训练 task：`General-Tracking-G1`
- 推理观测：`velcmd_history`（166D）
- ONNX 签名：dual-input `obs` + `obs_history`
- 训练采样模式：`uniform`
- playback / benchmark 采样模式：`start`
- 训练 `window_steps=[0]`
- actor/critic 使用较大的 TemporalCNN 配置（1024,512,256,256,128）
- realtime / offline reference window 配置入口：`retarget_buffer_enabled`、`retarget_buffer_window_s`、`retarget_buffer_delay_s`、`reference_steps`


## Dataset Flow

训练数据主线是：`typed source YAML -> preprocess/filter -> merged NPZ(train/val)`。

- dataset spec 支持 `preprocess` 段，用于 root xy 归一化、脚底对地和基础过滤
- dataset spec 支持 `window.reference_steps`，builder 会把有效采样范围写入 merged NPZ
- `MotionLib` 会按 `window_steps` 约束采样有效中心帧
- `window_steps=[0]` 保持当前主路径行为

## Constraints

- `controller.policy_path` 必须显式提供，且文件必须存在。
- `controller.observation_type` 必须与 ONNX 训练语义匹配；不会自动切换，也不会 pad/trim。
- 离线 BVH 运行必须显式传 `input.bvh_file=...`。
- viewer 只接受 `viewers` 配置键；旧 `viewer` alias 已移除。
- Pico4 依赖 `xrobotoolkit_sdk`，不会随 `pip install -e .` 自动安装。
- sim2sim 默认应使用 `g1_mjlab.xml`；`g1_mocap_29dof.xml` 仅用于运动学可视化。

## Docs

- [`docs/getting-started.md`](docs/getting-started.md)：最短上手路径
- [`docs/inference.md`](docs/inference.md)：离线/在线推理、viewer、录制、渲染
- [`docs/training.md`](docs/training.md)：训练、播放、benchmark、导出 ONNX
- [`docs/sim2real.md`](docs/sim2real.md)：Unitree G1 实机部署与状态机
- [`docs/dataset.md`](docs/dataset.md)：数据集构建主线
- [`docs/configuration.md`](docs/configuration.md)：Hydra 配置入口
- [`docs/architecture.md`](docs/architecture.md)：系统边界和装配关系
- [`docs/training_troubleshooting.md`](docs/training_troubleshooting.md)：训练问题排查
