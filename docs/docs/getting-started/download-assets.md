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
python scripts/setup/download_assets.py --only robots gmr ckpt bvh
```

## Asset Inventory

Downloaded file sizes change as checkpoints, datasets, and asset bundles are updated. Use the repository paths below as the stable contract.

| Local Path | Purpose |
|------------|---------|
| `track.onnx` | ONNX inference model |
| `track.pt` | PyTorch checkpoint for resume training |
| `data/datasets/seed/shard_*.h5` | Minimal motion dataset; run precompute before training |
| `data/sample_bvh/*.bvh` | Sample motion files |
| `assets/robots/unitree_g1/` | Canonical G1 XML and meshes used by training, sim2sim, retargeting, and FK validation |
| `teleopit/retargeting/gmr/assets/` | GMR retargeting assets, IK configs, and non-canonical robot descriptions |

## Asset Groups

| Group | ModelScope Repo | Contents |
|-------|----------------|----------|
| `ckpt` | `BingqianWu/Teleopit-models` | `track.onnx`, `track.pt` |
| `robots` | `BingqianWu/Teleopit-models` | Canonical robot XML/meshes |
| `gmr` | `BingqianWu/Teleopit-models` | GMR retargeting assets |
| `bvh` | `BingqianWu/Teleopit-models` | Sample BVH motion files |
| `data` | `BingqianWu/Teleopit-datasets` | Training/validation shards |

For asset management details (uploading, versioning), see [Asset Management](../reference/assets).
