# 数据集

## 下载预构建数据集（推荐）

从 ModelScope 下载已处理好的训练数据，可直接用于训练和评估：

```bash
# 下载训练数据
modelscope download --model BingqianWu/Teleopit --include "data/train/**" --include "data/val/**" --local_dir teleopit-assets

# 放到项目目录
mkdir -p data/datasets/seed
cp -r teleopit-assets/data/train data/datasets/seed/train
cp -r teleopit-assets/data/val data/datasets/seed/val
```

训练时直接传 shard 目录：

```bash
python train_mimic/scripts/train.py --motion_file data/datasets/seed/train
```

如需自定义构建数据集，继续阅读下文。

---

## 自定义构建

数据主线：`typed source YAML -> preprocess/filter -> shard-only 训练数据`

核心命令：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml
```

## 输出目录

每个 dataset 的全部正式产物都落在同一个目录：

```text
data/datasets/<dataset>/
├── clips/                  # 可选；仅在需要逐 clip 中间产物时存在
│   └── <source>/...
├── train/
│   └── shard_*.npz
├── val/
│   └── shard_*.npz
├── manifest_resolved.csv
└── build_info.json
```

这意味着导入、导出、备份一个 dataset 时，直接处理整个 `data/datasets/<dataset>/` 目录即可。

补充说明：

- 如果 spec 包含 `bvh` 或 `npz` source，builder 会保留/生成标准 `clips/`
- 如果 spec 全部是 `pkl` 或 `seed_csv` source，builder 会走 batch 路径，直接并行产出 split 级别的 shard 文件，默认不再写中间 clip 文件

## YAML spec

`train_mimic/configs/datasets/twist2_full.yaml` 示例：

```yaml
name: twist2_full
target_fps: 30
val_percent: 5
hash_salt: ""
preprocess:
  normalize_root_xy: true
  ground_align: clip_min_foot
sources:
  - name: OMOMO_g1_GMR
    type: pkl
    input: data/twist2_retarget_pkl/OMOMO_g1_GMR
  - name: lafan1_v1
    type: bvh
    input: data/lafan1_bvh
    bvh_format: lafan1
```

字段说明：

- `name`: dataset 名称，对应输出目录 `data/datasets/<name>/`
- `target_fps`: 写入 shard 前统一重采样到的目标帧率
- `val_percent`: 基于 `clip_id` hash 的验证集比例
- `hash_salt`: 可选 split salt
- `preprocess.normalize_root_xy`: 是否把根 body 首帧 `xy` 平移到原点
- `preprocess.ground_align`: `none` / `clip_min_foot` / `frame_min_foot`
- `preprocess.min_frames`: clip 最短长度约束
- `preprocess.max_root_lin_vel` / `min_peak_body_height` / `max_all_off_ground_s`: 基础过滤阈值
- `sources[].name`: source 名称；在会生成 clip 中间产物的路径里，它也会成为 `clips/<source>/` 子目录名
- `sources[].type`: `bvh` / `pkl` / `npz` / `seed_csv`
- `sources[].input`: 原始输入文件或目录。对于 `type: npz`，目录应直接指向 clip 目录，不要指向已有 dataset 根目录
- `sources[].weight`: 可选源级别采样权重，默认 `1.0`
- `sources[].bvh_format`: 仅 `bvh` source 必填，当前支持 `lafan1` / `hc_mocap` / `nokov`
- `sources[].robot_name`: 仅 `bvh` source 使用，默认 `unitree_g1`；当前数据集 BVH 转换只支持 `unitree_g1`
- `sources[].max_frames`: 仅 `bvh` source 使用，默认 `0` 表示全长

## 转换规则

统一转换目标是标准训练 shard；每个 clip 在写入 shard 前都会先经过 preprocess/filter；中间 clip 是否落盘取决于 source 类型组合：

- `bvh -> retarget pkl -> npz clip`
- `pkl -> npz clip`，或在 `pkl-only` dataset 中直接 batch 写入 split shard
- `npz -> validate + copy/reuse`

最终 build 阶段始终产出 shard 目录，不再支持单文件 `train.npz` / `val.npz`。

每个 shard 会写入：

- `clip_starts`
- `clip_lengths`
- `clip_fps`
- `clip_weights`

这些字段定义了 shard 内 clip 边界和采样权重。训练采样器会在运行时根据 `window_steps` 计算每个 clip 的有效中心帧范围。

## 常用命令

强制重建整个 dataset 目录：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --force
```

按文件多进程并行转换：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --jobs 8
```

输出到自定义 datasets 根目录：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --output_root /tmp/my_datasets
```

跳过 sampled FK check：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --skip_fk_check
```

打印 build report：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2_full.yaml \
    --json
```

## 单独批量转换为 NPZ clips

如果你只想先把某一批原始数据转成标准 NPZ clip，而不做 train/val merge，可以用：

```bash
python train_mimic/scripts/data/ingest_motion.py \
    --type bvh \
    --input data/lafan1_bvh \
    --output data/datasets/lafan1_v1/clips/lafan1_v1 \
    --source lafan1_v1 \
    --bvh_format lafan1 \
    --jobs 8
```

或者：

```bash
python train_mimic/scripts/data/ingest_motion.py \
    --type pkl \
    --input data/twist2_retarget_pkl/OMOMO_g1_GMR \
    --output data/datasets/omomo_only/clips/omomo \
    --source omomo \
    --jobs 8
```

## 单独检查 clip

检查某个 clip NPZ 的 FK 一致性：

```bash
python train_mimic/scripts/data/check_motion_npz_fk.py \
    --npz data/datasets/twist2_full/clips/<source>/<clip>.npz
```

如果当前 dataset 是 `pkl-only` batch build，默认不会有 `clips/<source>/<clip>.npz`；这时如需逐 clip 检查，请先用 `ingest_motion.py` 单独落盘该 source。

推荐判据：

- `pos_max < 1e-3 m`
- `quat_mean < 0.05 rad`
- `quat_p95 < 0.10 rad`

## 重新切分 shard（用于分发或大文件管理）

如果现有 shard 目录过大（如单个 shard >5G），可以重新切成更多 shard：

```bash
python train_mimic/scripts/data/split_shards.py \
    --input data/datasets/twist2_full/train \
    --output data/datasets/twist2_full/train_small_shards \
    --max_size_gb 2
```

每个 shard 是自包含的 merged NPZ（含完整 clip metadata），训练时直接传目录：

```bash
python train_mimic/scripts/train.py \
    --motion_file data/datasets/twist2_full/train_small_shards
```

## 和训练主线的连接

构建完成后，训练和评估直接消费 shard 目录：

```bash
python train_mimic/scripts/train.py \
    --motion_file data/datasets/twist2_full/train

python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/twist2_full/val \
    --num_envs 1
```
