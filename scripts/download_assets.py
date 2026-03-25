#!/usr/bin/env python3
"""One-click download script for Teleopit assets from ModelScope.

Usage:
    python scripts/download_assets.py            # download everything
    python scripts/download_assets.py --only gmr  # only GMR retargeting assets
    python scripts/download_assets.py --only ckpt  # only checkpoints
    python scripts/download_assets.py --only data  # only training data
    python scripts/download_assets.py --only bvh   # only sample BVH
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = "BingqianWu/Teleopit"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (remote_path, local_path_relative_to_project_root)
ASSET_GROUPS = {
    "ckpt": [
        ("checkpoints/track.onnx", "track.onnx"),
        ("checkpoints/track.pt", "track.pt"),
    ],
    "data": [
        ("data/train", "data/datasets/seed/train"),
        ("data/val", "data/datasets/seed/val"),
    ],
    "bvh": [
        ("data/sample_bvh/aiming1_subject1.bvh", "data/sample_bvh/aiming1_subject1.bvh"),
    ],
    "gmr": [
        ("gmr_assets", "teleopit/retargeting/gmr/assets"),
    ],
}


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def download_all(groups, cache_dir):
    """Download from ModelScope to a local cache, then copy to project paths."""
    try:
        from modelscope import snapshot_download  # noqa: F401
    except ImportError:
        print("modelscope not installed. Installing...")
        run([sys.executable, "-m", "pip", "install", "modelscope"])

    # Download entire repo to cache (ModelScope handles incremental caching)
    print(f"\nDownloading {REPO_ID} to {cache_dir} ...")
    run([
        sys.executable, "-m", "modelscope", "download",
        "--model", REPO_ID,
        "--local_dir", str(cache_dir),
    ])

    # Copy assets to their target locations
    for group in groups:
        entries = ASSET_GROUPS[group]
        print(f"\n[{group}] Placing files...")
        for remote_rel, local_rel in entries:
            src = cache_dir / remote_rel
            dst = PROJECT_ROOT / local_rel
            if not src.exists():
                print(f"  SKIP {remote_rel} (not found in download)")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"  {remote_rel} -> {local_rel}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Download Teleopit assets from ModelScope"
    )
    parser.add_argument(
        "--only",
        choices=list(ASSET_GROUPS.keys()),
        nargs="+",
        help="Only download specific asset groups (default: all)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "modelscope_cache"),
        help="Local cache directory for ModelScope download",
    )
    args = parser.parse_args()

    groups = args.only or list(ASSET_GROUPS.keys())
    download_all(groups, Path(args.cache_dir))


if __name__ == "__main__":
    main()
