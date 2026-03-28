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
import tarfile
from pathlib import Path

from teleopit.runtime.external_assets import ASSET_GROUPS, AssetEntry

REPO_ID = "BingqianWu/Teleopit-assets"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _safe_extract_tar(archive_path: Path, dst: Path) -> None:
    tmp_dst = dst.parent / f".{dst.name}.extracting"
    _remove_path(tmp_dst)
    tmp_dst.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe archive member path: {member.name}")
        tar.extractall(tmp_dst)

    _remove_path(dst)
    tmp_dst.replace(dst)


def _copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        _remove_path(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _resolve_entry_source(cache_dir: Path, entry: AssetEntry) -> Path | None:
    candidate = cache_dir / entry.remote_path
    if candidate.exists():
        return candidate
    return None


def download_all(groups, cache_dir):
    """Download from ModelScope to a local cache, then copy to project paths."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("modelscope not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
        from modelscope import snapshot_download

    print(f"\nDownloading {REPO_ID} to {cache_dir} ...")
    snapshot_download(REPO_ID, local_dir=str(cache_dir))

    # Copy assets to their target locations
    for group in groups:
        entries = ASSET_GROUPS[group]
        print(f"\n[{group}] Placing files...")
        for entry in entries:
            src = _resolve_entry_source(cache_dir, entry)
            local_rel = entry.local_path
            dst = PROJECT_ROOT / local_rel
            if src is None:
                print(f"  SKIP {entry.remote_path} (not found in download)")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if entry.mode == "extract" and src.is_file():
                _safe_extract_tar(src, dst)
                print(f"  {src.relative_to(cache_dir)} -> {local_rel} (extracted)")
            else:
                _copy_path(src, dst)
                print(f"  {src.relative_to(cache_dir)} -> {local_rel}")

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
