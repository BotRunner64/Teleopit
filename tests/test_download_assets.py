from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_assets import _resolve_entry_source, _safe_extract_tar
from scripts.prepare_modelscope_assets import _archive_directory
from teleopit.runtime.external_assets import AssetEntry


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
