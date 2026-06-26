---
sidebar_position: 2
---

# Asset Management

Datasets, checkpoints, robot models, and demo media are not tracked in Git. They are distributed via ModelScope and HuggingFace. The canonical Unitree G1 model is downloaded to `assets/robots/unitree_g1/g1_29dof.xml`.

## What's Not in Git

- `assets/robots/` - Canonical robot XML/meshes
- `teleopit/retargeting/gmr/assets/` - GMR retargeting assets, IK configs, and non-canonical robot descriptions
- `data/`, checkpoints, caches
- Demo media (`assets/demo.gif`, `assets/demo.mp4`)

## Repositories

### ModelScope (default download source)

| Repository | Type | Contents |
|-----------|------|----------|
| `BingqianWu/Teleopit-models` | model | Checkpoints, GMR retargeting assets, sample BVH |
| `BingqianWu/Teleopit-datasets` | dataset | Training/validation datasets |

### HuggingFace (alternative)

| Repository | Type | Contents |
|-----------|------|----------|
| `12e21/Teleopit-models` | model | Checkpoints, GMR retargeting assets, sample BVH |
| `12e21/Teleopit-datasets` | dataset | Training/validation datasets |

### Asset Group Mapping

| Group | Repository | Remote Path |
|-------|-----------|-------------|
| `ckpt` | Teleopit-models | `checkpoints/track.onnx`, `checkpoints/track.pt` |
| `robots` | Teleopit-models | `archives/robot_assets.tar.gz` |
| `gmr` | Teleopit-models | `archives/gmr_assets.tar.gz` |
| `bvh` | Teleopit-models | `archives/sample_bvh.tar.gz` |
| `data` | Teleopit-datasets | `data/datasets/*/*.h5` (`lafan1`, `pico_record`, `seed`, `twist2`) |

## Download

Use the project download script (defaults to ModelScope):

```bash
# Download everything
python scripts/setup/download_assets.py

# Only inference essentials
python scripts/setup/download_assets.py --only robots gmr ckpt bvh

# Only training data
python scripts/setup/download_assets.py --only data

# Download from HuggingFace instead
python scripts/setup/download_assets.py --source huggingface
```

Local paths after download:

| Remote | Local |
|--------|-------|
| `checkpoints/track.onnx` | `track.onnx` |
| `checkpoints/track.pt` | `track.pt` |
| `archives/robot_assets.tar.gz` | `assets/robots/` (extracted) |
| `archives/gmr_assets.tar.gz` | `teleopit/retargeting/gmr/assets/` (extracted) |
| `archives/sample_bvh.tar.gz` | `data/sample_bvh/` (extracted) |
| `data/datasets/*/*.h5` | `data/datasets/` |

## Upload to ModelScope

### Step 1: Prepare Upload Directory

```bash
python scripts/setup/prepare_modelscope_assets.py --only ckpt robots gmr bvh --clean
python scripts/setup/prepare_modelscope_assets.py --only data
```

Output goes to `data/modelscope_upload/`.

### Step 2: Upload

```bash
# Model repo
modelscope upload --repo-type model BingqianWu/Teleopit-models \
    data/modelscope_upload/checkpoints checkpoints
modelscope upload --repo-type model BingqianWu/Teleopit-models \
    data/modelscope_upload/archives archives

# Dataset repo
modelscope upload --repo-type dataset BingqianWu/Teleopit-datasets \
    data/modelscope_upload/data data
```

### Step 3: Tag Version

Only the model repo supports tags (dataset repo does not).

```bash
python - <<'EOF'
from modelscope.hub.api import HubApi
api = HubApi()
url = api.create_model_tag("BingqianWu/Teleopit-models", "vX.Y.Z")
print(url)
EOF
```

Tags should match Git tags for traceability.

## Upload to HuggingFace

### Step 1: Prepare and Upload

```bash
# Prepare and upload model assets (--clean ensures no leftover files)
python scripts/setup/upload_hf_assets.py --only ckpt robots gmr bvh --clean

# Prepare and upload dataset
python scripts/setup/upload_hf_assets.py --only data --clean
```

Use `--dry-run` to stage files locally without uploading.

:::warning
Always use `--clean` when running `--only`, otherwise the staging directory may carry leftover files from a previous run, causing unintended uploads.
:::

### Step 2: Tag Version

```bash
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_tag("12e21/Teleopit-models", tag="vX.Y.Z", repo_type="model")
EOF
```

## Pre-Push Check

```bash
python scripts/dev/check_large_tracked_files.py
```

This blocks large binary files and checks tracked file size limits.
