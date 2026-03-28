<p align="center">
  <img src="assets/teleopit_logo.jpg" width="80" alt="Teleopit">
</p>

<p align="center">
  <h1 align="center">Teleopit</h1>
  <h3 align="center">轻量、可扩展的人形机器人全身遥操作框架</h3>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="docs/pico4.md">Pico VR 部署</a> •
  <a href="docs/sim2real.md">真机部署</a> •
  <a href="docs/training.md">训练</a> •
  <a href="docs/assets.md">资源说明</a>
</p>

## 安装

```bash
pip install -e .              # 推理（sim2sim）
pip install -e '.[train]'     # 训练
pip install -e '.[sim2real]'  # 真机部署
```

## 下载模型和数据

一键下载所有资源（模型、数据、GMR retargeting assets）：

```bash
pip install modelscope
python scripts/download_assets.py
```

只下载推理必需的部分：

```bash
python scripts/download_assets.py --only gmr ckpt bvh
```

下载内容说明：

| 资源 | 大小 | 用途 |
|------|------|------|
| `track.onnx` | 4M | ONNX 推理模型 |
| `track.pt` | 27M | PyTorch checkpoint（resume 训练用） |
| `data/datasets/seed/train/shard_*.npz` | ~25G | 训练集 |
| `data/datasets/seed/val/shard_*.npz` | ~1.4G | 验证集 |
| `data/sample_bvh/*.bvh` | 5M | 示例动作文件 |
| `teleopit/retargeting/gmr/assets/` | ~1.2G | GMR retargeting 机器人模型 |

## Quick Start

离线 sim2sim：

```bash
python scripts/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

训练（传 shard 目录）：

```bash
python train_mimic/scripts/train.py \
    --motion_file data/datasets/seed/train \
    --num_envs 4096 \
    --max_iterations 30000
```

导出 ONNX：

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --output track.onnx \
    --history_length 10
```

## 更多用法

| 场景 | 命令 | 文档 |
|------|------|------|
| **Pico 4 VR 遥操作** | `python scripts/run_sim.py --config-name pico4_sim ...` | **[Pico VR 部署](docs/pico4.md)** |
| **Pico 4 真机部署** | `python scripts/run_sim2real.py --config-name pico4_sim2real ...` | **[Pico VR 部署](docs/pico4.md)** |
| UDP 实时 sim2sim | `python scripts/run_sim.py --config-name online ...` | [inference.md](docs/inference.md) |
| G1 真机部署（UDP） | `python scripts/run_sim2real.py ...` | [sim2real.md](docs/sim2real.md) |
| 训练与导出 | `python train_mimic/scripts/train.py ...` | [training.md](docs/training.md) |

## 文档

- **[Pico VR 部署](docs/pico4.md)**：Pico 4 / Pico 4 Ultra 全身追踪遥操作完整指南
- [真机部署](docs/sim2real.md)：Unitree G1 部署、状态机、UDP 输入
- [推理与运行](docs/inference.md)：离线/在线推理、viewer、录制
- [训练](docs/training.md)：训练、评估、导出 ONNX
- [数据集](docs/dataset.md)：数据下载与自定义构建
- [资源管理](docs/assets.md)：外部资源下载与 ModelScope 上传
- [配置说明](docs/configuration.md)：Hydra 配置入口
- [架构](docs/architecture.md)：系统边界与技术规格

## 更新日志

### v0.1.0 (2025-03-25)

首个公开版本：General-Tracking-G1 全身追踪训练、ONNX sim2sim 推理、Pico 4 VR 遥操作、Unitree G1 真机部署。
