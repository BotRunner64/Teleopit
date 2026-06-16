---
sidebar_position: 3
---

# Dataset

## Download Pre-Built Dataset (Recommended)

```bash
python scripts/setup/download_assets.py --only robots data
```

Then precompute the training shard and train with the precomputed dataset root:

```bash
python train_mimic/scripts/data/precompute_dataset.py \
    data/datasets/seed --outdir data/datasets/seed_precomputed --jobs 8
python train_mimic/scripts/train.py --motion_file data/datasets/seed_precomputed
```

For custom dataset construction, read on.

---

## Record Pico Clips

Use the interactive Pico recorder to create training-ready NPZ clips from live
body tracking:

```bash
pip install -e '.[pico4]'
python scripts/run/record_pico_motion.py
```

The recorder starts the Pico receiver and live `Retarget` viewer before waiting
for clip names, so preview keeps running while the terminal is idle. Enter a
semantic clip name, then use `R` to start, `S` to save, `D` to discard, `N` to
enter a new name, and `Q` to quit. Saved clips go to
`data/pico_motion/clips/` as `<semantic_label>_<timestamp>.npz`; no per-clip
JSON is written, so clips can be renamed or deleted manually.

Build all recorded clips into the standard HDF5 shard dataset:

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec data/pico_motion/pico_recorded.yaml --force
```

At least one valid clip is required after preprocessing.

## Custom Dataset Construction

Data pipeline: `typed source YAML -> preprocess/filter -> minimal HDF5 shards -> precomputed training dataset`

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec train_mimic/configs/datasets/twist2.yaml
```

## Output Structure

```text
data/datasets/<dataset>/
└── shard_*.h5

data/datasets/<dataset>_precomputed/
└── shard_*.h5
```

- If the spec contains `bvh` or `npz` sources, the full dataset builder uses a temporary `clips/` directory during conversion and deletes it after shards are written. Rebuilds do not reuse converted clips.
- If the spec is all `pkl` or `seed_csv` sources, the builder takes a batch path producing shards directly
- `build_dataset.py` only writes the minimal distributable dataset. It does not run FK precompute.
- `precompute_dataset.py` writes a separate training dataset containing the minimal motion plus precomputed joint velocities and body FK/velocities.
- Training accepts only the precomputed dataset directory. It recursively discovers precomputed `*.h5` shards below the specified root, so precomputed datasets can be merged by placing multiple shard directories under one parent.
- Training loads only a subset cache from the discovered precomputed shards, stages the next cache asynchronously, and swaps caches at the PPO rollout barrier. Joint velocities and body FK/velocities are not computed during training.

## YAML Spec Format

Example (`train_mimic/configs/datasets/twist2.yaml`):

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

### Field Reference

| Field | Description |
|-------|-------------|
| `name` | Dataset name, maps to output directory |
| `target_fps` | Target frame rate for resampling |
| `preprocess.normalize_root_xy` | Normalize root body first-frame xy to origin |
| `preprocess.ground_align` | `none` / `first_frame_foot` |
| `preprocess.min_frames` | Minimum clip length |
| `preprocess.max_root_lin_vel` | Root linear velocity filter threshold |
| `preprocess.min_peak_body_height` | Minimum peak body height |
| `preprocess.max_all_off_ground_s` | Max duration all feet off ground |
| `sources[].name` | Source name |
| `sources[].type` | `bvh` / `pkl` / `npz` / `seed_csv` |
| `sources[].input` | Input file or directory |
| `sources[].bvh_format` | Required for BVH: `lafan1` / `hc_mocap` / `nokov` |
| `sources[].robot_name` | BVH only, default `unitree_g1` |
| `sources[].max_frames` | BVH only, `0` = full length |

## Conversion Rules

All sources are converted to standard minimal shards. Each clip goes through preprocessing/filtering before writing to shards:

- `bvh -> retarget pkl -> npz clip`
- `pkl -> npz clip` (or direct batch shard for pkl-only datasets)
- `npz -> validate + copy/reuse`

Each minimal shard stores `root_pos`, `root_quat_w`, `joint_pos`, `body_names`, `clip_starts`, `clip_lengths`, and `clip_fps`. The precomputed training shards store `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, and the same metadata. Training fails fast if `--motion_file` points at a minimal dataset instead of a precomputed training dataset.

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

# Generate a precomputed training dataset from an existing minimal dataset
python train_mimic/scripts/data/precompute_dataset.py \
    data/datasets/twist2 --outdir data/datasets/twist2_precomputed --jobs 8 --force

# Inspect a dataset root
python train_mimic/scripts/data/inspect_dataset.py data/datasets/twist2
```

## Batch Ingest to NPZ Clips

Convert raw data to standard NPZ clips without merging:

```bash
python train_mimic/scripts/data/ingest_motion.py \
    --type bvh --input data/lafan1_bvh \
    --output data/lafan1_clips/lafan1 \
    --source lafan1 --bvh_format lafan1 --jobs 8
```

## Check Clip FK Consistency

```bash
python train_mimic/scripts/data/check_motion_npz_fk.py \
    --npz data/lafan1_clips/lafan1/<clip>.npz
```

Recommended thresholds: `pos_max < 1e-3 m`, `quat_mean < 0.05 rad`, `quat_p95 < 0.10 rad`.
