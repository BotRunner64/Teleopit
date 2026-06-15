---
sidebar_position: 2
---

# Download Assets

Robot models, datasets, and checkpoints are hosted on ModelScope and must be downloaded before use.

## One-Click Download

Download all assets (models, data, GMR retargeting assets):

```bash
pip install modelscope
python scripts/setup/download_assets.py
```

## Selective Download

Download only what you need for inference:

```bash
python scripts/setup/download_assets.py --only gmr ckpt bvh
```

## Asset Inventory

| Asset | Size | Purpose |
|-------|------|---------|
| `track.onnx` | 4 MB | ONNX inference model |
| `track.pt` | 27 MB | PyTorch checkpoint (for resume training) |
| `data/datasets/seed/shard_*.h5` | ~26 GB | Training dataset |
| `data/sample_bvh/*.bvh` | 5 MB | Sample motion files |
| `teleopit/retargeting/gmr/assets/` | ~1.2 GB | GMR retargeting robot models |

## Asset Groups

| Group | ModelScope Repo | Contents |
|-------|----------------|----------|
| `ckpt` | `BingqianWu/Teleopit-models` | `track.onnx`, `track.pt` |
| `gmr` | `BingqianWu/Teleopit-models` | GMR retargeting assets |
| `bvh` | `BingqianWu/Teleopit-models` | Sample BVH motion files |
| `data` | `BingqianWu/Teleopit-datasets` | Training/validation shards |

For asset management details (uploading, versioning), see [Asset Management](../reference/assets).
