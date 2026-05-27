#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE="modelscope"
REPO_ID=""
DEST="${PROJECT_ROOT}/third_party/somehand/assets/mjcf"
CACHE_DIR="${PROJECT_ROOT}/data/somehand_assets_cache"

usage() {
  cat <<'EOF'
Download somehand LinkerHand L6 bi-hand MJCF assets.

Usage:
  scripts/setup/download_somehand_l6_assets.sh
  scripts/setup/download_somehand_l6_assets.sh --source huggingface
  scripts/setup/download_somehand_l6_assets.sh --dest third_party/somehand/assets/mjcf

Options:
  --source modelscope|huggingface   Download backend (default: modelscope)
  --repo-id REPO                    Override asset repo id
  --dest PATH                       Destination mjcf directory
  --cache-dir PATH                  Download cache directory
  -h, --help                        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --dest)
      DEST="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${SOURCE}" != "modelscope" && "${SOURCE}" != "huggingface" ]]; then
  echo "--source must be modelscope or huggingface" >&2
  exit 2
fi

python - "$PROJECT_ROOT" "$SOURCE" "$REPO_ID" "$DEST" "$CACHE_DIR" <<'PY'
from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
source = sys.argv[2]
repo_id = sys.argv[3] or ("BingqianWu/somehand-assets" if source == "modelscope" else "12e21/somehand-assets")
dest = Path(sys.argv[4]).expanduser()
if not dest.is_absolute():
    dest = (project_root / dest).resolve()
cache_dir = Path(sys.argv[5]).expanduser()
if not cache_dir.is_absolute():
    cache_dir = (project_root / cache_dir).resolve()

hands = ("linkerhand_l6_left", "linkerhand_l6_right")
direct_patterns = [f"assets/mjcf/{hand}/**" for hand in hands]
archive_pattern = "archives/mjcf_assets.tar.gz"
repo_cache = cache_dir / source / repo_id.split("/")[-1]


def ensure_package(name: str):
    try:
        return __import__(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", name])
        return __import__(name)


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def copy_dir(src: Path, dst: Path) -> None:
    remove_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def download(patterns: list[str]) -> None:
    repo_cache.mkdir(parents=True, exist_ok=True)
    if source == "huggingface":
        hub = ensure_package("huggingface_hub")
        hub.snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(repo_cache),
            allow_patterns=patterns,
        )
        return
    modelscope = ensure_package("modelscope")
    modelscope.snapshot_download(
        repo_id,
        repo_type="model",
        local_dir=str(repo_cache),
        allow_patterns=patterns,
        allow_file_pattern=patterns,
    )


def place_direct_assets() -> bool:
    placed = False
    for hand in hands:
        src = repo_cache / "assets" / "mjcf" / hand
        if not src.exists():
            return False
        copy_dir(src, dest / hand)
        print(f"  {src.relative_to(repo_cache)} -> {dest / hand}")
        placed = True
    return placed


def safe_extract_l6_from_archive(archive_path: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    wanted_prefixes = {f"mjcf/{hand}/" for hand in hands}
    with tarfile.open(archive_path, "r:*") as tar:
        members = []
        for member in tar.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member path: {member.name}")
            normalized = member.name.lstrip("./")
            if any(normalized.startswith(prefix) for prefix in wanted_prefixes):
                members.append(member)
        if not members:
            raise FileNotFoundError(f"No LinkerHand L6 assets found in {archive_path}")
        tmp = dest.parent / ".somehand_l6_extracting"
        remove_path(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        tar.extractall(tmp, members=members)
        for hand in hands:
            src = tmp / "mjcf" / hand
            if not src.exists():
                raise FileNotFoundError(f"Archive missing mjcf/{hand}")
            copy_dir(src, dest / hand)
            print(f"  archive:{hand} -> {dest / hand}")
        remove_path(tmp)


print(f"Downloading somehand L6 assets from {source}:{repo_id}")
print(f"Destination: {dest}")

download(direct_patterns)
if not place_direct_assets():
    print("Direct L6 asset paths not found; downloading mjcf archive fallback.")
    download([archive_pattern])
    archive = repo_cache / archive_pattern
    if not archive.exists():
        raise FileNotFoundError(f"Downloaded repo is missing {archive_pattern}")
    safe_extract_l6_from_archive(archive)

for hand in hands:
    model = dest / hand / "model.xml"
    if not model.exists():
        raise FileNotFoundError(f"Expected model file not found: {model}")

print("Done.")
PY
