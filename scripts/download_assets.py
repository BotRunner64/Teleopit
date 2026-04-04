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

from teleopit.runtime.external_assets import (
    ASSET_GROUPS,
    AssetEntry,
    MODEL_REPO_ID,
    DATASET_REPO_ID,
)

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


def download_all(groups, cache_dir):
    """Download from ModelScope to a local cache, then copy to project paths."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("modelscope not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
        from modelscope import snapshot_download

    # Split entries by repo type
    entries_by_repo: dict[str, list[AssetEntry]] = {
        MODEL_REPO_ID: [],
        DATASET_REPO_ID: [],
    }
    repo_type_map = {MODEL_REPO_ID: "model", DATASET_REPO_ID: "dataset"}

    for group in groups:
        for entry in ASSET_GROUPS[group]:
            repo_id = MODEL_REPO_ID if entry.repo == "model" else DATASET_REPO_ID
            entries_by_repo[repo_id].append(entry)

    # Download from each repo separately
    all_entries = []
    for repo_id, repo_entries in entries_by_repo.items():
        if not repo_entries:
            continue
        repo_type = repo_type_map[repo_id]
        allow_patterns = [f"{e.remote_path}*" for e in repo_entries]
        repo_cache = cache_dir / repo_type / repo_id.split("/")[-1]

        print(f"\nDownloading {repo_id} ({repo_type}) to {repo_cache} ...")
        print(f"Fetching: {[e.remote_path for e in repo_entries]}")
        snapshot_download(
            repo_id,
            repo_type=repo_type,
            local_dir=str(repo_cache),
            allow_patterns=allow_patterns,
            allow_file_pattern=allow_patterns,
        )
        all_entries.extend((repo_cache, e) for e in repo_entries)

    # Copy assets to their target locations
    print("\nPlacing files...")
    for repo_cache, entry in all_entries:
        src = repo_cache / entry.remote_path if (repo_cache / entry.remote_path).exists() else None
        local_rel = entry.local_path
        dst = PROJECT_ROOT / local_rel
        if src is None:
            print(f"  SKIP {entry.remote_path} (not found in download)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if entry.mode == "extract" and src.is_file():
            _safe_extract_tar(src, dst)
            print(f"  {entry.remote_path} -> {local_rel} (extracted)")
        else:
            _copy_path(src, dst)
            print(f"  {entry.remote_path} -> {local_rel}")

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
