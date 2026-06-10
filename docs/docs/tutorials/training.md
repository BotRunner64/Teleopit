---
sidebar_position: 5
---

# Training

Train a whole-body tracking policy and export it as ONNX for inference.

:::info
For data preparation, see [Dataset Reference](../reference/dataset). For common training issues, see [Training Troubleshooting](../reference/training-troubleshooting).
:::

## Setup

```bash
conda create -n teleopit python=3.10
conda activate teleopit
pip install -e '.[train]'
```

Verify:
```bash
python -c "import train_mimic.tasks; print('training OK')"
```

## Training

### Smoke Test

```bash
python train_mimic/scripts/train.py \
    --num_envs 64 \
    --max_iterations 100 \
    --motion_file data/datasets/seed/train
```

### Full Training

```bash
python train_mimic/scripts/train.py \
    --num_envs 4096 \
    --max_iterations 30000 \
    --motion_file data/datasets/seed/train
```

### Multi-GPU

```bash
python train_mimic/scripts/train.py \
    --gpu_ids 0 1 2 3 \
    --num_envs 1024 \
    --max_iterations 30000 \
    --motion_file data/datasets/seed/train
```

### Multi-Node Multi-GPU

Use `torchrun` directly when training across multiple machines:

```bash
torchrun \
    --nnodes=$PET_NNODES \
    --nproc_per_node=$PET_NPROC_PER_NODE \
    --node_rank=$PET_NODE_RANK \
    --master_addr=$PET_MASTER_ADDR \
    --master_port=$PET_MASTER_PORT \
    train_mimic/scripts/train.py \
    --num_envs 1024 \
    --max_iterations 1000 \
    --motion_file data/datasets/seed/train
```

**Notes:**
- `--num_envs` is per-GPU in multi-GPU mode
- `--num_envs` is also per-process in multi-node mode, so total environments scale with `world_size`
- Default logger is TensorBoard. Use `--logger wandb` or `--logger swanlab` to select W&B or SwanLab; the project name defaults to `experiment_name`
- `--motion_file` accepts only shard directories (containing `shard_*.npz` files)
- `--max_iterations` means additional iterations; resuming from `model_12000.pt` with `--max_iterations 18000` trains to `model_30000.pt`

## Export ONNX

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --output track.onnx \
    --history_length 10
```

The exported model is a dual-input ONNX (`obs` + `obs_history`). The inference side only supports 166D dual-input ONNX.

## Evaluation

### Playback

```bash
python train_mimic/scripts/play.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/seed/val
```

### Benchmark

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/seed/val \
    --num_envs 1
```

### Benchmark with Video

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/seed/val \
    --num_envs 1 \
    --video \
    --video_length 600
```

## Training Architecture

```text
train_mimic/scripts
    -> train_mimic/app.py
    -> single task registry / env builder / runner cfg
    -> mjlab + rsl_rl
```

Key files:
- `train_mimic/app.py` - Shared entry point for train/play/benchmark
- `train_mimic/tasks/tracking/config/env.py` - General-Tracking-G1 env builder
- `train_mimic/tasks/tracking/config/rl.py` - TemporalCNN PPO config
- `train_mimic/tasks/tracking/mdp/commands.py` - Supports `uniform` / `start` sampling modes
