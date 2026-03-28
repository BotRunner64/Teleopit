from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetEntry:
    remote_path: str
    local_path: str
    mode: str = "copy"


ASSET_GROUPS: dict[str, list[AssetEntry]] = {
    "ckpt": [
        AssetEntry("checkpoints/track.onnx", "track.onnx"),
        AssetEntry("checkpoints/track.pt", "track.pt"),
    ],
    "data": [
        AssetEntry("data/train", "data/datasets/seed/train"),
        AssetEntry("data/val", "data/datasets/seed/val"),
    ],
    "bvh": [
        AssetEntry(
            "archives/sample_bvh.tar.gz",
            "data/sample_bvh",
            mode="extract",
        ),
    ],
    "gmr": [
        AssetEntry(
            "archives/gmr_assets.tar.gz",
            "teleopit/retargeting/gmr/assets",
            mode="extract",
        ),
    ],
}
