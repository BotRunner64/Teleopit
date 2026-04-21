---
sidebar_position: 3
---

# Dataset

## Download Pre-Built Dataset (Recommended)

```bash
python scripts/setup/download_assets.py --only data
```

Then train directly with the shard directory:

```bash
python train_mimic/scripts/train.py --motion_file data/datasets/seed/train
```

For custom dataset construction, read on.

---

## Custom Dataset Construction

Data pipeline: `typed source YAML -> preprocess/filter -> shard-only training data`

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml
```

## Output Structure

```text
data/datasets/<dataset>/
├── clips/                  # Optional; only for per-clip intermediates
│   └── <source>/...
├── train/
│   └── shard_*.npz
├── val/
│   └── shard_*.npz
├── manifest_resolved.csv
└── build_info.json
```

- If the spec contains `bvh` or `npz` sources, the builder retains/generates `clips/`
- If the spec is all `pkl` or `seed_csv` sources, the builder takes a batch path producing split-level shards directly

## YAML Spec Format

Example (`train_mimic/configs/datasets/twist2.yaml`):

```yaml
name: twist2
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
  - name: lafan1
    type: bvh
    input: data/lafan1_bvh
    bvh_format: lafan1
```

### Field Reference

| Field | Description |
|-------|-------------|
| `name` | Dataset name, maps to output directory |
| `target_fps` | Target frame rate for resampling |
| `val_percent` | Validation split percentage (hash-based on clip_id) |
| `hash_salt` | Optional split salt |
| `preprocess.normalize_root_xy` | Normalize root body first-frame xy to origin |
| `preprocess.ground_align` | `none` / `clip_min_foot` |
| `preprocess.min_frames` | Minimum clip length |
| `preprocess.max_root_lin_vel` | Root linear velocity filter threshold |
| `preprocess.min_peak_body_height` | Minimum peak body height |
| `preprocess.max_all_off_ground_s` | Max duration all feet off ground |
| `sources[].name` | Source name (used for clips subdirectory) |
| `sources[].type` | `bvh` / `pkl` / `npz` / `seed_csv` |
| `sources[].input` | Input file or directory |
| `sources[].weight` | Optional sampling weight (default `1.0`) |
| `sources[].bvh_format` | Required for BVH: `lafan1` / `hc_mocap` / `nokov` |
| `sources[].robot_name` | BVH only, default `unitree_g1` |
| `sources[].max_frames` | BVH only, `0` = full length |

## Conversion Rules

All sources are converted to standard training shards. Each clip goes through preprocess/filter before writing to shards:

- `bvh -> retarget pkl -> npz clip`
- `pkl -> npz clip` (or direct batch shard for pkl-only datasets)
- `npz -> validate + copy/reuse`

Each shard contains: `clip_starts`, `clip_lengths`, `clip_fps`, `clip_weights`.

## Common Commands

```bash
# Force rebuild
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml --force

# Parallel processing
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml --jobs 8

# Custom output root
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml \
    --output_root /tmp/my_datasets

# Print build report
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml --json
```

## Batch Ingest to NPZ Clips

Convert raw data to standard NPZ clips without merging:

```bash
python train_mimic/scripts/data/ingest_motion.py \
    --type bvh --input data/lafan1_bvh \
    --output data/datasets/lafan1/clips/lafan1 \
    --source lafan1 --bvh_format lafan1 --jobs 8
```

## Check Clip FK Consistency

```bash
python train_mimic/scripts/data/check_motion_npz_fk.py \
    --npz data/datasets/<dataset>/clips/<source>/<clip>.npz
```

Recommended thresholds: `pos_max < 1e-3 m`, `quat_mean < 0.05 rad`, `quat_p95 < 0.10 rad`.

## Re-shard

Split large shards for distribution:

```bash
python train_mimic/scripts/data/split_shards.py \
    --input data/datasets/seed/train \
    --output data/datasets/seed/train_small_shards \
    --max_size_gb 2
```

Each shard is self-contained with full clip metadata.
