# 训练指南

> 数据准备看 [`dataset.md`](dataset.md)，架构细节看 [`architecture.md`](architecture.md)。

## 环境安装

```bash
conda create -n teleopit python=3.10
conda activate teleopit

cd Teleopit
pip install -e '.[train]'
```

快速校验：

```bash
python -c "import train_mimic.tasks; print('training OK')"
```

## 训练

smoke test：

```bash
python train_mimic/scripts/train.py \
    --num_envs 64 \
    --max_iterations 100 \
    --motion_file data/datasets/twist2_full/train
```

完整训练：

```bash
python train_mimic/scripts/train.py \
    --num_envs 4096 \
    --max_iterations 30000 \
    --motion_file data/datasets/twist2_full/train
```

单机多卡：

```bash
python train_mimic/scripts/train.py \
    --gpu_ids 0 1 2 3 \
    --num_envs 1024 \
    --max_iterations 30000 \
    --motion_file data/datasets/twist2_full/train
```

说明：

- `--num_envs` 在多卡模式下表示每张卡的环境数
- 默认 logger 是 tensorboard；传 `--wandb_project <name>` 才启用 wandb
- `--motion_file` 只接受 shard 目录（内含一个或多个 `shard_*.npz`）
- `--max_iterations` 表示追加训练轮数；resume `model_12000.pt` 且传 `--max_iterations 18000` 会训练到 `model_30000.pt`
- 默认实验名是 `g1_general_tracking`

## 导出 ONNX

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --output policy.onnx \
    --history_length 10
```

导出结果约束：

- 导出 `TemporalCNN` checkpoint 为双输入 ONNX（`obs` + `obs_history`）
- 推理侧只支持 166D 双输入 ONNX

## 播放与评估

playback：

```bash
python train_mimic/scripts/play.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/twist2_full/val
```

benchmark：

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/twist2_full/val \
    --num_envs 1
```

benchmark 录视频：

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/twist2_full/val \
    --num_envs 1 \
    --video \
    --video_length 600
```

## 训练侧边界

```text
train_mimic/scripts
    -> train_mimic/app.py
    -> single task registry / env builder / runner cfg
    -> mjlab + rsl_rl

train_mimic/scripts/data
    -> dataset_builder
    -> dataset_lib / motion_fk / convert_pkl_to_npz
```

关键实现：

- `train_mimic/app.py`：train/play/benchmark 共用入口装配
- `train_mimic/tasks/tracking/config/env.py`：General-Tracking-G1 env builder
- `train_mimic/tasks/tracking/config/rl.py`：TemporalCNN PPO 配置
- `train_mimic/tasks/tracking/mdp/commands.py`：支持 `adaptive` / `uniform` / `start` 采样模式
