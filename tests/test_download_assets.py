from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.setup.download_assets import (
    _clear_cached_entry_sources,
    _entry_allow_patterns,
    _resolve_entry_source,
    _safe_extract_tar,
)
from scripts.setup.prepare_modelscope_assets import _archive_directory
from teleopit.runtime.external_assets import ASSET_GROUPS, AssetEntry


def test_archive_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("hello\n", encoding="utf-8")
    (src / "nested").mkdir()
    (src / "nested" / "b.txt").write_text("world\n", encoding="utf-8")

    archive = tmp_path / "bundle.tar.gz"
    _archive_directory(src, archive)

    dst = tmp_path / "dst"
    _safe_extract_tar(archive, dst)

    assert (dst / "a.txt").read_text(encoding="utf-8") == "hello\n"
    assert (dst / "nested" / "b.txt").read_text(encoding="utf-8") == "world\n"

def test_resolve_entry_source_uses_only_current_remote_layout(tmp_path: Path) -> None:
    archive = tmp_path / "archives" / "gmr_assets.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"archive")

    entry = AssetEntry(
        remote_path="archives/gmr_assets.tar.gz",
        local_path="teleopit/retargeting/gmr/assets",
        mode="extract",
    )

    assert _resolve_entry_source(tmp_path, entry) == archive


def test_robot_asset_group_uses_archive_layout() -> None:
    entries = ASSET_GROUPS["robots"]

    assert len(entries) == 1
    assert entries[0].remote_path == "archives/robot_assets.tar.gz"
    assert entries[0].local_path == "assets/robots"
    assert entries[0].mode == "extract"


def test_data_asset_group_downloads_only_hdf5_shards() -> None:
    entries = ASSET_GROUPS["data"]

    assert len(entries) == 1
    assert entries[0].remote_path == "data/datasets"
    assert entries[0].local_path == "data/datasets"
    assert _entry_allow_patterns(entries[0]) == ["data/datasets/*/*.h5"]


def test_clear_cached_entry_sources_removes_stale_data_files(tmp_path: Path) -> None:
    entry = ASSET_GROUPS["data"][0]
    dataset_cache = tmp_path / "data" / "datasets"
    dataset_cache.mkdir(parents=True)
    (dataset_cache / "old_clip.npz").write_bytes(b"old")
    (dataset_cache / "shard_000.h5").write_bytes(b"new")

    _clear_cached_entry_sources(tmp_path, [entry])

    assert not dataset_cache.exists()
