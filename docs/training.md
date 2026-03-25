# 训练指南

当前 `train_mimic` 只保留一个训练任务：`General-Tracking-G1`。

对应约束：

- 训练脚本不再需要 `--task` 参数；默认且唯一的任务是 `General-Tracking-G1`
- policy 结构使用 `TemporalCNNModel`
- 导出的 ONNX 是双输入 `obs` + `obs_history`

> 入口导航：数据准备看 [`dataset.md`](dataset.md)，整体边界看 [`architecture.md`](architecture.md)。

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

## 数据集准备

推荐入口：

```bash
python train_mimic/scripts/data/build_dataset.py             --spec train_mimic/configs/datasets/twist2_full.yaml
```

常用变体：

```bash
python train_mimic/scripts/data/build_dataset.py             --spec train_mimic/configs/datasets/twist2_full.yaml             --force

python train_mimic/scripts/data/build_dataset.py             --spec train_mimic/configs/datasets/twist2_full.yaml             --jobs 4
```

构建产物：

- `data/datasets/twist2_full/train.npz`
- `data/datasets/twist2_full/val.npz`
- `data/datasets/twist2_full/build_info.json`
- `data/datasets/twist2_full/manifest_resolved.csv`

FK 一致性检查：

```bash
python train_mimic/scripts/data/check_motion_npz_fk.py             --npz data/datasets/twist2_full/clips/<source>/<clip>.npz
```

## 训练

smoke test：

```bash
python train_mimic/scripts/train.py             --num_envs 64             --max_iterations 100             --motion_file data/datasets/twist2_full/train.npz
```

完整训练：

```bash
python train_mimic/scripts/train.py             --num_envs 4096             --max_iterations 30000             --motion_file data/datasets/twist2_full/train.npz
```

单机多卡：

```bash
python train_mimic/scripts/train.py             --gpu_ids 0 1 2 3             --num_envs 1024             --max_iterations 30000             --motion_file data/datasets/twist2_full/train.npz
```

说明：

- `--num_envs` 在多卡模式下表示每张卡的环境数
- 默认 logger 是 tensorboard；传 `--wandb_project <name>` 才启用 wandb
- `--motion_file` 必须指向单个 merged NPZ
- `--max_iterations` 表示追加训练轮数；resume `model_12000.pt` 且传 `--max_iterations 18000` 会训练到 `model_30000.pt`
- 默认实验名是 `g1_general_tracking`

## 导出 ONNX

```bash
python train_mimic/scripts/save_onnx.py             --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt             --output policy.onnx             --history_length 10
```

导出结果约束：

- 导出 `TemporalCNN` checkpoint 为双输入 ONNX（`obs` + `obs_history`）
- 推理侧只支持 166D 双输入 ONNX

## 播放与评估

playback：

```bash
python train_mimic/scripts/play.py             --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt             --motion_file data/datasets/twist2_full/val.npz
```

benchmark：

```bash
python train_mimic/scripts/benchmark.py             --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt             --motion_file data/datasets/twist2_full/val.npz             --num_envs 1
```

benchmark 录视频：

```bash
python train_mimic/scripts/benchmark.py             --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt             --motion_file data/datasets/twist2_full/val.npz             --num_envs 1             --video             --video_length 600
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
