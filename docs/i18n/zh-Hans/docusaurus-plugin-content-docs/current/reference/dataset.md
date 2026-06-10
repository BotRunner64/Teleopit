---
sidebar_position: 3
---

# 数据集

## 下载预构建数据集（推荐）

```bash
python scripts/setup/download_assets.py --only data
```

下载后直接传 shard 目录用于训练：

```bash
python train_mimic/scripts/train.py --motion_file data/datasets/seed/train
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

将所有已录制 clips 构建为标准 shard 数据集：

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec data/pico_motion/pico_recorded.yaml --force
```

构建前至少录制两段 clip，确保 train 和 validation split 都能生成。

## 自定义构建

数据主线：`typed source YAML -> preprocess/filter -> shard-only 训练数据`

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml
```

## 输出目录结构

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

- 若 spec 包含 `bvh` 或 `npz` source，builder 会保留/生成 `clips/`
- 若 spec 全部是 `pkl` 或 `seed_csv` source，直接并行产出 split 级别的 shard，默认不写中间 clip 文件

## YAML spec

示例（`train_mimic/configs/datasets/twist2.yaml`）：

```yaml
name: twist2
target_fps: 30
val_percent: 5
hash_salt: ""
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
| `val_percent` | 基于 `clip_id` hash 的验证集比例 |
| `hash_salt` | 可选 split salt |
| `preprocess.normalize_root_xy` | 是否把根 body 首帧 xy 平移到原点 |
| `preprocess.ground_align` | `none` / `first_frame_foot` |
| `preprocess.min_frames` | clip 最短长度约束 |
| `preprocess.max_root_lin_vel` / `min_peak_body_height` / `max_all_off_ground_s` | 基础过滤阈值 |
| `sources[].name` | source 名称；生成 clip 中间产物时也作为 `clips/<source>/` 子目录名 |
| `sources[].type` | `bvh` / `pkl` / `npz` / `seed_csv` |
| `sources[].input` | 原始输入文件或目录 |
| `sources[].weight` | 可选源级别采样权重，默认 `1.0` |
| `sources[].bvh_format` | 仅 `bvh` source 必填：`lafan1` / `hc_mocap` / `nokov` |
| `sources[].robot_name` | 仅 `bvh` source，默认 `unitree_g1` |
| `sources[].max_frames` | 仅 `bvh` source，`0` 表示全长 |

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
```

## 批量转换为 NPZ clips

只把某批原始数据转成标准 NPZ clip，不做 train/val merge：

```bash
python train_mimic/scripts/data/ingest_motion.py \
    --type bvh --input data/lafan1_bvh \
    --output data/datasets/lafan1/clips/lafan1 \
    --source lafan1 --bvh_format lafan1 --jobs 8
```

## FK 一致性检查

```bash
python train_mimic/scripts/data/check_motion_npz_fk.py \
    --npz data/datasets/<dataset>/clips/<source>/<clip>.npz
```

推荐判据：`pos_max < 1e-3 m`、`quat_mean < 0.05 rad`、`quat_p95 < 0.10 rad`。

## 重新切分 shard

```bash
python train_mimic/scripts/data/split_shards.py \
    --input data/datasets/seed/train \
    --output data/datasets/seed/train_small_shards \
    --max_size_gb 2
```

每个 shard 是自包含的 merged NPZ（含完整 clip metadata），训练时直接传目录。
