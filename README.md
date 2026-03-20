# Teleopit

Teleopit 是一个轻量、可扩展、自包含的人形机器人全身遥操作框架。

当前仓库收敛到两条明确主线：

- 默认主线是 `Tracking-Flat-G1-VelCmdHistory`，对应 `velcmd_history` 166D 观测和 dual-input TemporalCNN ONNX（`obs` + `obs_history`）
- 新增可选主线是 `Tracking-Flat-G1-MotionTrackingDeploy`，它按兄弟仓库 `motion_tracking/sim2real` 的最终部署 policy 语义做了单阶段训练对齐，对应 `motion_tracking_deploy` 1590D 观测和 single-input MLP ONNX（`obs`）
- 默认运行配置仍然保持在安全的 `VelCmdHistory` 路径；deploy-aligned motion tracking 通过独立 Hydra config 显式启用
- teacher-student、多阶段训练和其他历史 task 变体不再是本仓库支持面的一部分

## Architecture

```text
InputProvider (BVH file / UDP realtime / Pico4)
    -> Retargeter (GMR)
    -> ObservationBuilder (166D VelCmdHistory or 1590D MotionTrackingDeploy)
    -> Controller (dual-input TemporalCNN ONNX or single-input MLP ONNX)
    -> Robot (MuJoCo sim or Unitree G1)
```

离线/在线推理由 `teleopit/runtime/` 和 `teleopit/pipeline.py` 装配，真机状态机保留在 `teleopit/sim2real/controller.py`。训练闭环由 `train_mimic/` 提供。

在线 realtime 路径会先把 retarget 后的 `qpos` 写入短时 reference timeline，再按控制时刻从 timeline 采样 reference window。默认配置仍使用 `reference_steps=[0]`；deploy-aligned motion tracking 配置则显式使用 `[0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16]`。非零 `reference_steps` 需要同时满足 `retarget_buffer_delay_s >= max_future_step / policy_hz`，以及 `retarget_buffer_window_s >= retarget_buffer_delay_s + abs(min_history_step) / policy_hz`，否则运行时会直接报错，不会静默 fallback。

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

默认 VelCmdHistory 离线 sim2sim：

```bash
python scripts/run_sim.py   controller.policy_path=policy.onnx   input.bvh_file=data/lafan1/dance1_subject2.bvh
```

默认 VelCmdHistory UDP 实时 sim2sim：

```bash
# Terminal 1
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx

# Terminal 2
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop
```

默认 VelCmdHistory Pico4 sim2sim：

```bash
python scripts/run_sim.py --config-name pico4_sim controller.policy_path=policy.onnx
```

默认 VelCmdHistory G1 sim2real：

```bash
python scripts/run_sim2real.py controller.policy_path=policy.onnx
```

deploy-aligned motion tracking 离线 sim2sim：

```bash
python scripts/run_sim.py   --config-name motion_tracking   controller.policy_path=policy.onnx   input.bvh_file=data/lafan1/dance1_subject2.bvh
```

deploy-aligned motion tracking UDP 实时 sim2sim：

```bash
python scripts/run_sim.py --config-name motion_tracking_online controller.policy_path=policy.onnx
```

deploy-aligned motion tracking Pico4 sim2sim / sim2real：

```bash
python scripts/run_sim.py --config-name motion_tracking_pico4_sim controller.policy_path=policy.onnx
python scripts/run_sim2real.py --config-name motion_tracking_pico4_sim2real controller.policy_path=policy.onnx
```

deploy-aligned motion tracking G1 sim2real：

```bash
python scripts/run_sim2real.py --config-name motion_tracking_sim2real controller.policy_path=policy.onnx
```

训练：

```bash
python train_mimic/scripts/train.py   --motion_file data/datasets/twist2_full/train.npz

python train_mimic/scripts/train.py   --task Tracking-Flat-G1-MotionTrackingDeploy   --motion_file data/datasets/twist2_full/train.npz
```

导出 ONNX：

```bash
# VelCmdHistory TemporalCNN
python train_mimic/scripts/save_onnx.py   --checkpoint logs/rsl_rl/g1_tracking_velcmd_history/<run>/model_30000.pt   --output policy.onnx   --history_length 10

# MotionTrackingDeploy MLP
python train_mimic/scripts/save_onnx.py   --checkpoint logs/rsl_rl/g1_tracking_motion_tracking_deploy/<run>/model_30000.pt   --output policy.onnx
```

`save_onnx.py` 会按 checkpoint 内容自动识别是 dual-input TemporalCNN 还是 single-input deploy MLP。

## Supported Surface

- 训练 task：`Tracking-Flat-G1-VelCmdHistory`、`Tracking-Flat-G1-MotionTrackingDeploy`
- 推理观测：`velcmd_history`（166D）和 `motion_tracking_deploy`（1590D）
- ONNX 签名：支持 dual-input `obs` + `obs_history`，也支持 single-input `obs`
- 默认训练 / 推理入口仍是 VelCmdHistory
- deploy-aligned motion tracking 必须通过 `teleopit/configs/controller/motion_tracking_policy.yaml` 和 `teleopit/configs/motion_tracking*.yaml` 显式启用
- 训练采样模式：`uniform`
- playback / benchmark 采样模式：`start`
- VelCmdHistory 默认训练 `window_steps=[0]`
- MotionTrackingDeploy 使用部署版 future/history window、单阶段 PPO 和 MLP actor/critic
- realtime / offline reference window 配置入口：`retarget_buffer_enabled`、`retarget_buffer_window_s`、`retarget_buffer_delay_s`、`reference_steps`

## Deploy-Aligned Motion Tracking Semantics

`motion_tracking_deploy` 对齐的是兄弟仓库 `motion_tracking/sim2real` 中实际部署的 final policy，而不是旧的训练侧近似版本：

- 观测模块顺序固定为 `BootIndicator -> TrackingCommandObsRaw -> ComplianceFlagObs -> TargetJointPosObs -> TargetRootZObs -> TargetProjectedGravityBObs -> RootAngVelBHistory -> ProjectedGravityBHistory -> JointPos -> JointVel -> PrevActions`
- 总维度固定为 `1590`
- 关键配置固定为 `future_steps=[0,1,2,3,4,-1,-2,-4,-8,-12,-16]`
- 历史步长固定为 `[0,1,2,3,4,8,12,16,20]`
- `prev_action_steps=8`
- `compliance_flag_value=1.0`，`compliance_flag_threshold=10.0`
- 运行时 builder 要求真实 `reference_window`，缺失时直接报错，不会静默复制当前 `qpos` 伪造窗口
- 运行时 builder 不依赖 `base_pos` / `base_lin_vel`，避免真机 `RobotState` 字段缺失时把 deploy 语义喂偏

## Dataset Flow

训练数据主线是：`typed source YAML -> preprocess/filter -> merged NPZ(train/val)`。

- dataset spec 支持 `preprocess` 段，用于 root xy 归一化、脚底对地和基础过滤
- dataset spec 支持 `window.reference_steps`，builder 会把有效采样范围写入 merged NPZ
- `MotionLib` 会按 `window_steps` 约束采样有效中心帧
- `window_steps=[0]` 保持当前 VelCmdHistory 主路径行为；deploy task 则使用多步 future/history window

## Constraints

- `controller.policy_path` 必须显式提供，且文件必须存在。
- `controller.observation_type` 必须与 ONNX 训练语义匹配；不会自动切换，也不会 pad/trim。
- `motion_tracking_deploy` 运行时要求匹配的 `reference_steps` 和真实 `reference_window`。
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
