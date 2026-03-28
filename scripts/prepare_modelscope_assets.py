#!/usr/bin/env python3
"""Prepare a compact upload directory for the ModelScope asset repo."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from teleopit.runtime.external_assets import ASSET_GROUPS, AssetEntry


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ModelScope upload files from local Teleopit assets."
    )
    parser.add_argument(
        "--only",
        choices=list(ASSET_GROUPS.keys()),
        nargs="+",
        help="Only prepare specific asset groups (default: all).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "modelscope_upload",
        help="Directory where the compact upload layout will be written.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing output directory before writing new files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    groups = args.only or list(ASSET_GROUPS.keys())

    if args.clean:
        _remove_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing ModelScope upload bundle in {output_dir}")
    for group in groups:
        print(f"\n[{group}]")
        for entry in ASSET_GROUPS[group]:
            _place_entry(output_dir, entry)
            print(f"  {entry.local_path} -> {entry.remote_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
