from __future__ import annotations

from dataclasses import dataclass, field


MODEL_REPO_ID = "BingqianWu/Teleopit-models"
DATASET_REPO_ID = "BingqianWu/Teleopit-datasets"

HF_MODEL_REPO_ID = "12e21/Teleopit-models"
HF_DATASET_REPO_ID = "12e21/Teleopit-datasets"


@dataclass(frozen=True)
class AssetEntry:
    remote_path: str
    local_path: str
    repo: str = "model"   # "model" or "dataset"
    mode: str = "copy"


ASSET_GROUPS: dict[str, list[AssetEntry]] = {
    "ckpt": [
        AssetEntry("checkpoints/track.onnx", "track.onnx", repo="model"),
        AssetEntry("checkpoints/track.pt", "track.pt", repo="model"),
    ],
    "gmr": [
        AssetEntry(
            "archives/gmr_assets.tar.gz",
            "teleopit/retargeting/gmr/assets",
            repo="model",
            mode="extract",
        ),
    ],
    "bvh": [
        AssetEntry(
            "archives/sample_bvh.tar.gz",
            "data/sample_bvh",
            repo="model",
            mode="extract",
        ),
    ],
    "data": [
        AssetEntry("data/train", "data/datasets/seed/train", repo="dataset"),
        AssetEntry("data/val", "data/datasets/seed/val", repo="dataset"),
    ],
}
