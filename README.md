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
  <a href="docs/training.md">训练</a>
</p>

<p align="center">
  <img src="assets/demo.gif" width="360" alt="Teleopit Demo">
</p>

## 安装

```bash
pip install -e .              # 推理（sim2sim）
pip install -e '.[train]'     # 训练
pip install -e '.[sim2real]'  # 真机部署
```

## 下载模型和数据

从 ModelScope 下载预训练模型和示例数据：

```bash
pip install modelscope
modelscope download --model BingqianWu/Teleopit --local_dir teleopit-assets
```

将下载的文件放到项目对应位置：

```bash
# checkpoint（推理用）
cp teleopit-assets/checkpoints/track.onnx .

# 示例 BVH（sim2sim 演示用）
mkdir -p data/sample_bvh
cp teleopit-assets/data/sample_bvh/aiming1_subject1.bvh data/sample_bvh/

# 训练数据（shard 目录，可直接传给 --motion_file）
mkdir -p data/datasets/seed
cp -r teleopit-assets/data/train data/datasets/seed/train
cp -r teleopit-assets/data/val data/datasets/seed/val
```

下载内容说明：

| 文件 | 用途 |
|------|------|
| `checkpoints/track.onnx` | ONNX 推理模型 |
| `checkpoints/track.pt` | PyTorch checkpoint（resume 训练用） |
| `data/train/shard_*.npz` | 训练集（shard 格式，~25G 总量） |
| `data/val/shard_*.npz` | 验证集（~1.4G） |
| `data/sample_bvh/aiming1_subject1.bvh` | 示例动作文件 |

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
- [配置说明](docs/configuration.md)：Hydra 配置入口
- [架构](docs/architecture.md)：系统边界与技术规格

## 更新日志

### v0.1.0 (2025-03-25)

首个公开版本：General-Tracking-G1 全身追踪训练、ONNX sim2sim 推理、Pico 4 VR 遥操作、Unitree G1 真机部署。
