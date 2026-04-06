#!/usr/bin/env python3
"""Prepare and upload Teleopit assets to HuggingFace repos.

Asset groups by destination repo:
  model   (12e21/Teleopit-models):   ckpt, gmr, bvh
  dataset (12e21/Teleopit-datasets): data

Usage:
    python scripts/setup/upload_hf_assets.py             # upload everything
    python scripts/setup/upload_hf_assets.py --only gmr  # only GMR assets
    python scripts/setup/upload_hf_assets.py --dry-run   # prepare files without uploading
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from teleopit.runtime.external_assets import (
    ASSET_GROUPS,
    AssetEntry,
    HF_MODEL_REPO_ID,
    HF_DATASET_REPO_ID,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        _remove_path(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _archive_directory(src: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_path(archive_path)
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(src.rglob("*")):
            tar.add(path, arcname=path.relative_to(src), recursive=False)


def _place_entry(output_dir: Path, entry: AssetEntry) -> None:
    src = PROJECT_ROOT / entry.local_path
    if not src.exists():
        raise FileNotFoundError(f"Local asset source not found: {src}")

    dst = output_dir / entry.remote_path
    if entry.mode == "extract":
        if not src.is_dir():
            raise ValueError(f"Archive-backed asset must be a directory: {src}")
        _archive_directory(src, dst)
    else:
        _copy_path(src, dst)


def _prepare_staging(groups: list[str], staging_dir: Path) -> dict[str, Path]:
    """Stage files into per-repo subdirectories. Returns {repo_id: staging_path} for touched repos only."""
    all_repo_staging: dict[str, Path] = {
        HF_MODEL_REPO_ID: staging_dir / "model",
        HF_DATASET_REPO_ID: staging_dir / "dataset",
    }

    # Determine which repos are actually needed for this run.
    touched_repos: set[str] = set()
    for group in groups:
        for entry in ASSET_GROUPS[group]:
            repo_id = HF_MODEL_REPO_ID if entry.repo == "model" else HF_DATASET_REPO_ID
            touched_repos.add(repo_id)

    # Clear only the touched repo subdirs so that --only <group> never carries
    # leftover files from a previous run into the upload.
    for repo_id in touched_repos:
        _remove_path(all_repo_staging[repo_id])
        all_repo_staging[repo_id].mkdir(parents=True, exist_ok=True)

    print(f"Staging files in {staging_dir}")
    for group in groups:
        print(f"\n[{group}]")
        for entry in ASSET_GROUPS[group]:
            repo_id = HF_MODEL_REPO_ID if entry.repo == "model" else HF_DATASET_REPO_ID
            out_dir = all_repo_staging[repo_id]
            _place_entry(out_dir, entry)
            print(f"  {entry.local_path} -> {entry.remote_path}")

    # Return only the repos that were staged; _upload must not see untouched dirs.
    return {repo_id: all_repo_staging[repo_id] for repo_id in touched_repos}


def _upload(repo_staging: dict[str, Path]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    repo_type_map = {HF_MODEL_REPO_ID: "model", HF_DATASET_REPO_ID: "dataset"}

    for repo_id, staging_path in repo_staging.items():
        # Check if there are actually files staged for this repo
        if not any(staging_path.rglob("*")):
            continue

        repo_type = repo_type_map[repo_id]
        print(f"\nCreating repo {repo_id} ({repo_type}) if not exists...")
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

        print(f"Uploading {staging_path} -> {repo_id} ...")
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(staging_path),
            commit_message="Upload Teleopit assets",
        )
        print(f"  Done: https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and upload Teleopit assets to HuggingFace."
    )
    parser.add_argument(
        "--only",
        choices=list(ASSET_GROUPS.keys()),
        nargs="+",
        help="Only upload specific asset groups (default: all).",
    )
    parser.add_argument(
        "--staging_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "hf_upload_staging",
        help="Temporary staging directory for prepared files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing staging directory before preparing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare files locally but skip the actual upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    staging_dir = args.staging_dir.expanduser().resolve()
    groups = args.only or list(ASSET_GROUPS.keys())

    if args.clean:
        _remove_path(staging_dir)

    repo_staging = _prepare_staging(groups, staging_dir)

    if args.dry_run:
        print("\nDry-run: skipping upload.")
    else:
        _upload(repo_staging)

    print("\nDone!")


if __name__ == "__main__":
    main()
