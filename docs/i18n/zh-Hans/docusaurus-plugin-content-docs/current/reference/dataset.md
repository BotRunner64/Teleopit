---
sidebar_position: 3
---

# 数据集

## 下载预构建数据集（推荐）

```bash
python scripts/setup/download_assets.py --only robots data
```

下载后先预计算所有已下载数据集，再把合并后的预计算数据集根目录用于训练：

```bash
python train_mimic/scripts/data/precompute_dataset.py \
    data/datasets --outdir data/datasets_precomputed --jobs 8
python train_mimic/scripts/train.py --motion_file data/datasets_precomputed
```

如需自定义构建，继续阅读下文。

---

## 录制 Pico clips

使用交互式 Pico 录制脚本，从实时 body tracking 生成训练可用的 NPZ clips：

```bash
pip install -e '.[pico4]'
python scripts/run/record_pico_motion.py
```

录制器会先启动 Pico receiver 和实时 `Retarget` viewer，再等待输入 clip 名；
因此终端空闲时预览仍会持续运行。输入动作语义名后，用 `R` 开始录制、`S`
保存、`D` 丢弃、`N` 输入新名字、`Q` 退出。保存的 clip 会写入
`data/pico_motion/clips/`，文件名格式为 `<semantic_label>_<timestamp>.npz`；不会写
每段 clip 的 JSON，因此可以手动改名或删除。

将所有已录制 clips 构建为标准 HDF5 shard 数据集：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec data/pico_motion/pico_recorded.yaml --force
```

预处理后至少需要保留一段有效 clip。

## 自定义构建

数据主线：`typed source YAML -> preprocess/filter -> minimal HDF5 shards -> precomputed training dataset`

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml
```

## 输出目录结构

```text
data/datasets/<dataset>/
└── shard_*.h5

data/datasets_precomputed/<dataset>/
└── shard_*.h5
```

- 若 spec 包含 `bvh` 或 `npz` source，完整 dataset builder 会在转换期间使用临时 `clips/` 目录，并在 shard 写入完成后删除。重新 build 不会复用已转换 clips。
- 若 spec 全部是 `pkl` 或 `seed_csv` source，builder 会直接并行产出 shard，默认不写中间 clip 文件
- `build_dataset.py` 只写最小分发数据集，不执行 FK 预计算。
- `precompute_dataset.py` 会写出独立的训练数据集，里面包含最小运动数据以及预计算的 joint velocity 和 body FK/velocity。
- 训练只接受预计算后的数据集目录。它会递归发现指定根目录下的预计算 `*.h5` shard，因此使用 `data/datasets_precomputed` 可以一起训练所有已下载数据集。
- 训练会在启动时把所有发现的预计算 motion window 全量加载到内存中。joint velocity 和 body FK/velocity 不会在训练时计算。

## YAML spec

示例（`train_mimic/configs/datasets/twist2.yaml`）：

```yaml
name: twist2
target_fps: 30
preprocess:
  normalize_root_xy: true
  ground_align: first_frame_foot
sources:
  - name: OMOMO_g1_GMR
    type: pkl
    input: data/twist2_retarget_pkl/OMOMO_g1_GMR
  - name: lafan1
    type: bvh
    input: data/lafan1_bvh
    bvh_format: lafan1
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `name` | 数据集名称，对应输出目录 `data/datasets/<name>/` |
| `target_fps` | 写入 shard 前统一重采样到的目标帧率 |
| `preprocess.normalize_root_xy` | 是否把根 body 首帧 xy 平移到原点 |
| `preprocess.ground_align` | `none` / `first_frame_foot` |
| `preprocess.min_frames` | clip 最短长度约束 |
| `preprocess.max_root_lin_vel` / `min_peak_body_height` / `max_all_off_ground_s` | 基础过滤阈值 |
| `sources[].name` | source 名称 |
| `sources[].type` | `bvh` / `pkl` / `npz` / `seed_csv` |
| `sources[].input` | 原始输入文件或目录 |
| `sources[].bvh_format` | 仅 `bvh` source 必填：`lafan1` / `hc_mocap` / `nokov` |
| `sources[].robot_name` | 仅 `bvh` source，默认 `unitree_g1` |
| `sources[].max_frames` | 仅 `bvh` source，`0` 表示全长 |

## 转换规则

所有 source 都会转换成标准最小 shard。每段 clip 会先经过预处理/过滤，再写入 shard：

- `bvh -> retarget pkl -> npz clip`
- `pkl -> npz clip`（或在 pkl-only 数据集中直接 batch 写 shard）
- `npz -> validate + copy/reuse`

每个最小 shard 保存 `root_pos`、`root_quat_w`、`joint_pos`、`body_names`、`clip_starts`、`clip_lengths` 和 `clip_fps`。预计算训练 shard 保存 `joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w` 以及相同的元数据。如果 `--motion_file` 指向最小数据集而不是预计算训练数据集，训练会立即报错。

## 常用命令

```bash
# 强制重建
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml --force

# 多进程并行
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml --jobs 8

# 自定义输出根目录
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml \
    --output_root /tmp/my_datasets

# 打印 build report
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml --json

# 从所有已下载最小数据集生成合并后的预计算训练数据集
python train_mimic/scripts/data/precompute_dataset.py \
    data/datasets --outdir data/datasets_precomputed --jobs 8 --force

# 查看数据集统计
python train_mimic/scripts/data/inspect_dataset.py data/datasets/twist2
```

## 批量转换为 NPZ clips

只把某批原始数据转成标准 NPZ clip，不合并为 shard：

```bash
python train_mimic/scripts/data/ingest_motion.py \
    --type bvh --input data/lafan1_bvh \
    --output data/lafan1_clips/lafan1 \
    --source lafan1 --bvh_format lafan1 --jobs 8
```

## FK 一致性检查

```bash
python train_mimic/scripts/data/check_motion_npz_fk.py \
    --npz data/lafan1_clips/lafan1/<clip>.npz
```

推荐判据：`pos_max < 1e-3 m`、`quat_mean < 0.05 rad`、`quat_p95 < 0.10 rad`。
