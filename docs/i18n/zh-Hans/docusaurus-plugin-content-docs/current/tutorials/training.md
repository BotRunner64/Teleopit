---
sidebar_position: 5
---

# 训练

训练全身追踪策略，并导出为 ONNX 格式用于推理部署。

:::info
数据准备请参阅 [数据集参考](../reference/dataset)。常见训练问题请参阅 [训练故障排查](../reference/training-troubleshooting)。
:::

## 环境安装

```bash
conda create -n teleopit python=3.10
conda activate teleopit
pip install -e '.[train]'
```

验证安装：
```bash
python -c "import train_mimic.tasks; print('training OK')"
```

## 训练

### 冒烟测试

```bash
python train_mimic/scripts/train.py \
    --num_envs 64 \
    --max_iterations 100 \
    --motion_file data/datasets/seed/train
```

### 完整训练

```bash
python train_mimic/scripts/train.py \
    --num_envs 4096 \
    --max_iterations 30000 \
    --motion_file data/datasets/seed/train
```

### 多卡训练

```bash
python train_mimic/scripts/train.py \
    --gpu_ids 0 1 2 3 \
    --num_envs 1024 \
    --max_iterations 30000 \
    --motion_file data/datasets/seed/train
```

**注意事项：**
- 多卡模式下 `--num_envs` 为每张 GPU 的环境数量
- 默认日志工具为 TensorBoard；传入 `--wandb_project <name>` 可启用 W&B
- `--motion_file` 仅接受分片目录（包含 `shard_*.npz` 文件的目录）
- `--max_iterations` 表示追加迭代次数；例如从 `model_12000.pt` 恢复训练并设置 `--max_iterations 18000`，最终将训练到 `model_30000.pt`

## 导出 ONNX

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --output track.onnx \
    --history_length 10
```

导出的模型为双输入 ONNX（`obs` + `obs_history`）。推理端仅支持 166D 双输入 ONNX 格式。

## 评估

### 播放验证

```bash
python train_mimic/scripts/play.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/seed/val
```

### 定量评估

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/seed/val \
    --num_envs 1
```

### 带视频的定量评估

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/seed/val \
    --num_envs 1 \
    --video \
    --video_length 600
```

## 训练架构

```text
train_mimic/scripts
    -> train_mimic/app.py
    -> single task registry / env builder / runner cfg
    -> mjlab + rsl_rl
```

关键文件：
- `train_mimic/app.py` - 训练/播放/评估的统一入口
- `train_mimic/tasks/tracking/config/env.py` - General-Tracking-G1 环境构建器
- `train_mimic/tasks/tracking/config/rl.py` - TemporalCNN PPO 配置
- `train_mimic/tasks/tracking/mdp/commands.py` - 支持 `adaptive` / `uniform` / `start` 三种采样模式
