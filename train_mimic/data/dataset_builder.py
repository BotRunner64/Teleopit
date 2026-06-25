from __future__ import annotations

import csv
import fnmatch
import json
import os
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mujoco
import yaml

import numpy as np

from train_mimic.data.dataset_lib import (
    DEFAULT_HDF5_MAX_WINDOW_FRAMES,
    DEFAULT_HDF5_WINDOW_OVERLAP_FRAMES,
    FULL_CLIP_ARRAY_KEYS,
    inspect_clip_dict,
    inspect_npz,
    merge_npz_files,
    resample_along_time,
    write_hdf5_motion_shard,
    write_json,
)
from train_mimic.data.preprocess import (
    DatasetPreprocessSpec,
    preprocess_clip_dict,
    validate_preprocess_spec,
)
from train_mimic.data.motion_fk import MotionFkExtractor, compute_npz_fk_consistency
from train_mimic.scripts.convert_pkl_to_npz import (
    convert_pkl_to_arrays,
    convert_pkl_to_npz,
    convert_seed_csv_to_arrays,
)
from teleopit.retargeting.export_pkl import convert_bvh_to_retarget_pkl, mocap_xml_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = PROJECT_ROOT / "train_mimic" / "configs" / "datasets" / "twist2.yaml"
DEFAULT_DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"
DEFAULT_FK_SAMPLE_CLIPS = 2
DEFAULT_FK_SAMPLE_FRAMES = 16
SOURCE_TYPES = {"bvh", "pkl", "npz", "seed_csv"}
BVH_FORMATS = {"lafan1", "hc_mocap", "nokov"}
SUPPORTED_BVH_ROBOT_NAME = "unitree_g1"

_SOURCE_SUFFIXES = {
    "bvh": ".bvh",
    "pkl": ".pkl",
    "npz": ".npz",
    "seed_csv": ".csv",
}
_DATASET_ROOT_MARKERS = {
    "clips",
}

_PROCESS_FK_EXTRACTOR: MotionFkExtractor | None = None
_PROCESS_BVH_MODEL_CACHE: dict[str, mujoco.MjModel] = {}


def default_jobs() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8))


DEFAULT_JOBS = default_jobs()


@dataclass(frozen=True)
class DatasetSourceSpec:
    name: str
    type: str
    input: str
    bvh_format: str | None = None
    robot_name: str = "unitree_g1"
    max_frames: int = 0
    metadata_csv: str | None = None
    filters: dict[str, list] | None = None
    seed_filter_preset: str | None = None
    exclude_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    target_fps: int
    sources: list[DatasetSourceSpec]
    preprocess: DatasetPreprocessSpec = field(default_factory=DatasetPreprocessSpec)


@dataclass(frozen=True)
class DatasetClipRow:
    clip_id: str
    source: str
    file_rel: str
    num_frames: int
    fps: int
    resolved_npz_path: str
    clip_index: int = -1  # index into source clip metadata; -1 = standalone clip


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    clips_root: Path


@dataclass(frozen=True)
class FkCheckSummary:
    source: str
    clip: str
    pos_max: float
    quat_mean: float
    quat_p95: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceInputFile:
    path: Path
    rel_no_suffix: Path


@dataclass(frozen=True)
class ConversionTask:
    source_name: str
    source_type: str
    input_path: str
    output_path: str
    bvh_format: str | None = None
    robot_name: str = "unitree_g1"
    max_frames: int = 0
    mocap_xml: str | None = None
    preprocess: DatasetPreprocessSpec = field(default_factory=DatasetPreprocessSpec)


@dataclass(frozen=True)
class FilteredClipResult:
    input_path: str
    reason: str


_FILTERED_MARKER_SUFFIX = ".filtered.json"


@dataclass(frozen=True)
class SeedFilterRule:
    columns: tuple[str, ...]
    patterns: tuple[str, ...]
    label: str


_SEED_FILTER_PRESETS: dict[str, tuple[SeedFilterRule, ...]] = {
    "groot_strict": (
        SeedFilterRule(
            columns=("content_body_position",),
            patterns=("sitting", "on all fours", "handstand"),
            label="content_body_position",
        ),
        SeedFilterRule(
            columns=("content_type_of_movement",),
            patterns=("crawling", "on hands and knees", "rolling", "flipping", "climbing"),
            label="content_type_of_movement",
        ),
        SeedFilterRule(
            columns=("content_props",),
            patterns=("chair", "crutch", "crutches", "ladder", "box", "table", "bike", "scooter", "bed"),
            label="content_props",
        ),
        SeedFilterRule(
            columns=("filename", "move_name"),
            patterns=("safety_roll", "cartwheel", "box_jump", "monkey_jump", "walking_on_edge"),
            label="filename_or_move_name",
        ),
    )
}


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _validate_source_type(raw_type: object, spec_path: Path, source_name: str) -> str:
    source_type = str(raw_type or "").strip()
    if source_type not in SOURCE_TYPES:
        raise ValueError(
            f"source {source_name!r} has invalid type {source_type!r} in {spec_path}. "
            f"Expected one of {sorted(SOURCE_TYPES)}."
        )
    return source_type


def _validate_seed_filter_preset(
    source_type: str,
    seed_filter_preset: str | None,
    metadata_csv: str | None,
    *,
    spec_path: Path,
    source_name: str,
) -> str | None:
    if seed_filter_preset is None:
        return None
    if source_type != "seed_csv":
        raise ValueError(
            f"source {source_name!r} uses seed_filter_preset={seed_filter_preset!r}, "
            "but seed_filter_preset is supported only for seed_csv sources"
        )
    if metadata_csv is None:
        raise ValueError(
            f"source {source_name!r} uses seed_filter_preset={seed_filter_preset!r} "
            f"without metadata_csv: {spec_path}"
        )
    if seed_filter_preset not in _SEED_FILTER_PRESETS:
        raise ValueError(
            f"source {source_name!r} has unknown seed_filter_preset {seed_filter_preset!r} "
            f"in {spec_path}. Expected one of {sorted(_SEED_FILTER_PRESETS)}."
        )
    return seed_filter_preset


def _load_preprocess_spec(raw: object, spec_path: Path) -> DatasetPreprocessSpec:
    if raw is None:
        return DatasetPreprocessSpec()
    if not isinstance(raw, dict):
        raise ValueError(f"dataset spec preprocess must be a mapping: {spec_path}")
    foot_body_names = raw.get("foot_body_names", ("left_ankle_roll_link", "right_ankle_roll_link"))
    if isinstance(foot_body_names, list):
        foot_body_names = tuple(str(name) for name in foot_body_names)
    spec = DatasetPreprocessSpec(
        normalize_root_xy=bool(raw.get("normalize_root_xy", False)),
        root_body_name=str(raw.get("root_body_name", "pelvis")).strip() or "pelvis",
        ground_align=str(raw.get("ground_align", "none")).strip() or "none",
        foot_body_names=tuple(foot_body_names),
        min_frames=int(raw.get("min_frames", 1)),
        max_root_lin_vel=(
            None if raw.get("max_root_lin_vel") in (None, "", "null") else float(raw["max_root_lin_vel"])
        ),
        min_peak_body_height=(
            None
            if raw.get("min_peak_body_height") in (None, "", "null")
            else float(raw["min_peak_body_height"])
        ),
        max_all_off_ground_s=(
            None
            if raw.get("max_all_off_ground_s") in (None, "", "null")
            else float(raw["max_all_off_ground_s"])
        ),
        off_ground_height=float(raw.get("off_ground_height", 0.2)),
        max_feet_off_ground_s=(
            None
            if raw.get("max_feet_off_ground_s") in (None, "", "null")
            else float(raw["max_feet_off_ground_s"])
        ),
        foot_off_ground_height=float(raw.get("foot_off_ground_height", 0.08)),
    )
    return validate_preprocess_spec(spec)


def _load_exclude_patterns(raw: object, spec_path: Path, source_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"source {source_name!r} exclude_patterns must be a list in {spec_path}"
        )
    patterns = tuple(str(item).strip() for item in raw if str(item).strip())
    if not patterns:
        raise ValueError(
            f"source {source_name!r} exclude_patterns must contain at least one non-empty pattern"
        )
    return patterns


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(f"dataset spec not found: {spec_path}")
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset spec must be a mapping: {spec_path}")

    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError(f"dataset spec missing non-empty name: {spec_path}")

    target_fps = int(payload.get("target_fps", 0))
    if target_fps <= 0:
        raise ValueError(f"dataset spec target_fps must be > 0: {spec_path}")

    preprocess = _load_preprocess_spec(payload.get("preprocess"), spec_path)
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"dataset spec must define a non-empty sources list: {spec_path}")

    sources: list[DatasetSourceSpec] = []
    seen_names: set[str] = set()
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise ValueError(f"each source entry must be a mapping: {spec_path}")

        source_name = str(raw.get("name", "")).strip()
        if not source_name:
            raise ValueError(f"source entry missing non-empty name: {spec_path}")
        if source_name in seen_names:
            raise ValueError(f"duplicate source name in spec: {source_name}")
        seen_names.add(source_name)

        source_type = _validate_source_type(raw.get("type"), spec_path, source_name)
        source_input = str(raw.get("input", "")).strip()
        if not source_input:
            raise ValueError(f"source {source_name!r} missing non-empty input: {spec_path}")

        bvh_format = raw.get("bvh_format")
        if source_type == "bvh":
            bvh_format = str(bvh_format or "").strip()
            if bvh_format not in BVH_FORMATS:
                raise ValueError(
                    f"source {source_name!r} requires bvh_format in {sorted(BVH_FORMATS)}: {spec_path}"
                )
        else:
            bvh_format = None

        robot_name = str(raw.get("robot_name", "unitree_g1")).strip() or "unitree_g1"
        max_frames = int(raw.get("max_frames", 0))
        if max_frames < 0:
            raise ValueError(f"source {source_name!r} has negative max_frames: {max_frames}")

        metadata_csv = raw.get("metadata_csv")
        if metadata_csv is not None:
            metadata_csv = str(metadata_csv).strip() or None
        filters = raw.get("filters")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError(f"source {source_name!r} filters must be a mapping: {spec_path}")
        seed_filter_preset = raw.get("seed_filter_preset")
        if seed_filter_preset is not None:
            seed_filter_preset = str(seed_filter_preset).strip() or None
        seed_filter_preset = _validate_seed_filter_preset(
            source_type,
            seed_filter_preset,
            metadata_csv,
            spec_path=spec_path,
            source_name=source_name,
        )
        exclude_patterns = _load_exclude_patterns(
            raw.get("exclude_patterns"),
            spec_path,
            source_name,
        )

        sources.append(
            DatasetSourceSpec(
                name=source_name,
                type=source_type,
                input=source_input,
                bvh_format=bvh_format,
                robot_name=robot_name,
                max_frames=max_frames,
                metadata_csv=metadata_csv,
                filters=filters,
                seed_filter_preset=seed_filter_preset,
                exclude_patterns=exclude_patterns,
            )
        )

    return DatasetSpec(
        name=name,
        target_fps=target_fps,
        sources=sources,
        preprocess=preprocess,
    )


def resolve_dataset_paths(spec: DatasetSpec, *, output_root: str | Path | None = None) -> DatasetPaths:
    base_root = DEFAULT_DATASETS_ROOT if output_root is None else Path(output_root).expanduser().resolve()
    dataset_dir = base_root / spec.name
    clips_root = dataset_dir / "clips"
    return DatasetPaths(dataset_dir=dataset_dir, clips_root=clips_root)


def shard_output_path(dataset_dir: Path, shard_index: int) -> Path:
    return dataset_dir / f"shard_{shard_index:03d}.h5"


def _clear_existing_motion_shards(dataset_dir: Path) -> None:
    """Remove stale top-level HDF5 outputs before writing a fresh dataset build."""
    if not dataset_dir.exists():
        return
    for pattern in ("shard_*.h5", ".*_chunk_*.h5"):
        for path in dataset_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _clear_intermediate_clips(clips_root: Path) -> None:
    if not clips_root.exists() and not clips_root.is_symlink():
        return
    if clips_root.is_dir() and not clips_root.is_symlink():
        shutil.rmtree(clips_root)
    else:
        clips_root.unlink()


def _path_contains(path: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(path)
    except ValueError:
        return False
    return True


def _source_input_candidate_path(source: DatasetSourceSpec) -> Path:
    candidate = Path(source.input).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=False)


def _ensure_source_inputs_do_not_overlap_intermediate_clips(
    spec: DatasetSpec,
    clips_root: Path,
) -> None:
    clips_path = clips_root.resolve(strict=False)
    conflicts: list[tuple[str, Path]] = []
    for source in spec.sources:
        input_path = _source_input_candidate_path(source)
        if _path_contains(clips_path, input_path) or _path_contains(input_path, clips_path):
            conflicts.append((source.name, input_path))

    if not conflicts:
        return

    details = ", ".join(f"{name}={path}" for name, path in conflicts)
    raise ValueError(
        f"source input overlaps the temporary clips directory {clips_path}: {details}. "
        "The dataset builder deletes that directory during full builds, so move source clips "
        "outside data/datasets/<dataset>/clips or choose a different output root."
    )


def resolve_source_input_path(source: DatasetSourceSpec) -> Path:
    candidate = Path(source.input).expanduser()
    input_path = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"source input not found for {source.name}: {input_path}")
    return input_path


def _ensure_not_dataset_root_npz_input(source: DatasetSourceSpec, input_path: Path) -> None:
    if source.type != "npz" or not input_path.is_dir():
        return
    has_clips_dir = (input_path / "clips").is_dir()
    has_dataset_outputs = any((input_path / name).exists() for name in _DATASET_ROOT_MARKERS)
    if has_clips_dir and has_dataset_outputs:
        raise ValueError(
            f"npz source {source.name!r} points at dataset root {input_path}, which contains merged "
            "dataset outputs. Point the source at the clip directory instead, for example "
            f"{input_path / 'clips'}, or at a specific .npz clip file."
        )


def _resolve_bvh_xml_path(source: DatasetSourceSpec) -> Path:
    assert source.type == "bvh"
    if source.robot_name != SUPPORTED_BVH_ROBOT_NAME:
        raise ValueError(
            f"source {source.name!r} uses robot_name={source.robot_name!r}, but dataset BVH conversion "
            f"currently supports only {SUPPORTED_BVH_ROBOT_NAME!r}. Use robot_name={SUPPORTED_BVH_ROBOT_NAME!r}."
        )
    xml_path = mocap_xml_path(PROJECT_ROOT, SUPPORTED_BVH_ROBOT_NAME)
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found for BVH conversion: {xml_path}")
    return xml_path


def _filter_seed_csv_by_metadata(
    source: DatasetSourceSpec,
    all_files: list[SourceInputFile],
    input_dir: Path,
    *,
    quiet: bool = False,
    report: dict[str, Any] | None = None,
) -> tuple[list[SourceInputFile], dict[str, Any]]:
    """Filter seed_csv files using metadata_csv + filters from the source spec."""
    if report is None:
        report = {
            "source": source.name,
            "type": source.type,
            "metadata_csv": source.metadata_csv,
            "seed_filter_preset": source.seed_filter_preset,
            "exclude_patterns": list(source.exclude_patterns),
            "scanned_files": len(all_files),
            "metadata_rows_matched": len(all_files),
            "preset_rejected_rows": 0,
            "path_rejected_files": 0,
            "kept_files": len(all_files),
            "filtered_files": 0,
            "preset_reject_reasons": {},
            "path_reject_reasons": {},
        }
    if source.metadata_csv is None or (source.filters is None and source.seed_filter_preset is None):
        return all_files, report

    meta_path = Path(source.metadata_csv).expanduser()
    if not meta_path.is_absolute():
        meta_path = (PROJECT_ROOT / meta_path).resolve()
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata_csv not found for {source.name}: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if source.filters is not None:
            for col in source.filters:
                if col not in fieldnames:
                    raise ValueError(
                        f"filter column {col!r} not found in metadata CSV for {source.name}. "
                        f"Available: {sorted(fieldnames)}"
                    )
        if "move_g1_path" not in fieldnames:
            raise ValueError(f"metadata CSV missing move_g1_path column for {source.name}")
        if source.seed_filter_preset is not None:
            for rule in _SEED_FILTER_PRESETS[source.seed_filter_preset]:
                for col in rule.columns:
                    if col not in fieldnames:
                        raise ValueError(
                            f"seed_filter_preset column {col!r} not found in metadata CSV for "
                            f"{source.name}. Available: {sorted(fieldnames)}"
                        )

        # Normalize filter values to strings for comparison
        str_filters: dict[str, set[str]] = {}
        for col, allowed_values in (source.filters or {}).items():
            str_filters[col] = {str(v) for v in allowed_values}

        rows = []
        for row in reader:
            if all(row.get(col, "") in vals for col, vals in str_filters.items()):
                rows.append(row)
    report["metadata_csv"] = str(meta_path)
    report["metadata_rows_matched"] = len(rows)

    if source.seed_filter_preset is not None:
        reject_counts: Counter[str] = Counter()
        kept_rows = []
        for row in rows:
            reasons: list[str] = []
            for rule in _SEED_FILTER_PRESETS[source.seed_filter_preset]:
                for pattern in rule.patterns:
                    if any(pattern in row.get(col, "").lower() for col in rule.columns):
                        reasons.append(f"{rule.label}:{pattern}")
                        break
            if reasons:
                reject_counts.update(reasons)
                continue
            kept_rows.append(row)
        report["preset_rejected_rows"] = len(rows) - len(kept_rows)
        report["preset_reject_reasons"] = dict(sorted(reject_counts.items()))
        rows = kept_rows

    # Build set of allowed relative paths (without .csv suffix)
    allowed_rels: set[str] = set()
    for row in rows:
        g1_path = row["move_g1_path"]
        # move_g1_path is like "g1/csv/240918/body_check_001__A548.csv"
        # input_dir is e.g. data/SEED/g1/csv
        # Resolve g1_path relative to the SEED root (input_dir's grandparent)
        try:
            seed_root = input_dir.parent.parent  # data/SEED
            full_path = seed_root / g1_path
            rel_to_input = full_path.relative_to(input_dir).with_suffix("")
            allowed_rels.add(rel_to_input.as_posix())
        except ValueError:
            allowed_rels.add(Path(g1_path).stem)

    filtered = [f for f in all_files if f.rel_no_suffix.as_posix() in allowed_rels]
    report["kept_files"] = len(filtered)
    report["filtered_files"] = int(report.get("scanned_files", len(all_files))) - len(filtered)
    if not quiet:
        print(
            f"[FILTER] source={source.name}: {len(filtered)}/{len(all_files)} files "
            f"after metadata filtering"
        )
        if source.seed_filter_preset is not None and report["preset_rejected_rows"] > 0:
            print(
                f"[FILTER] source={source.name}: preset={source.seed_filter_preset} "
                f"rejected={report['preset_rejected_rows']} "
                f"reasons={report['preset_reject_reasons']}"
            )
    return filtered, report


def _collect_source_files_with_report(
    source: DatasetSourceSpec,
    *,
    quiet: bool = False,
) -> tuple[list[SourceInputFile], Path, dict[str, Any]]:
    input_path = resolve_source_input_path(source)
    _ensure_not_dataset_root_npz_input(source, input_path)
    suffix = _SOURCE_SUFFIXES[source.type]

    def _base_report(items_count: int) -> dict[str, Any]:
        return {
            "source": source.name,
            "type": source.type,
            "metadata_csv": source.metadata_csv,
            "seed_filter_preset": source.seed_filter_preset,
            "exclude_patterns": list(source.exclude_patterns),
            "scanned_files": items_count,
            "metadata_rows_matched": items_count,
            "preset_rejected_rows": 0,
            "path_rejected_files": 0,
            "kept_files": items_count,
            "filtered_files": 0,
            "preset_reject_reasons": {},
            "path_reject_reasons": {},
        }

    def _matches_exclude(item: SourceInputFile) -> str | None:
        rel_no_suffix = item.rel_no_suffix.as_posix()
        candidates = (
            rel_no_suffix,
            f"{rel_no_suffix}{suffix}",
            item.path.name,
            item.path.stem,
        )
        for pattern in source.exclude_patterns:
            pat = pattern.lower()
            for candidate in candidates:
                if fnmatch.fnmatchcase(candidate.lower(), pat):
                    return pattern
        return None

    def _apply_path_excludes(
        items: list[SourceInputFile],
        report: dict[str, Any],
    ) -> list[SourceInputFile]:
        if not source.exclude_patterns:
            return items
        reject_counts: Counter[str] = Counter()
        kept: list[SourceInputFile] = []
        for item in items:
            reason = _matches_exclude(item)
            if reason is None:
                kept.append(item)
            else:
                reject_counts[reason] += 1
        report["path_rejected_files"] = len(items) - len(kept)
        report["path_reject_reasons"] = dict(sorted(reject_counts.items()))
        report["kept_files"] = len(kept)
        report["filtered_files"] = len(items) - len(kept)
        if not quiet and report["path_rejected_files"] > 0:
            print(
                f"[FILTER] source={source.name}: path_excludes rejected="
                f"{report['path_rejected_files']} reasons={report['path_reject_reasons']}"
            )
        if not kept:
            raise ValueError(
                f"no files remain after path exclude filtering for source {source.name}: {input_path}"
            )
        return kept

    if input_path.is_file():
        if input_path.suffix.lower() != suffix:
            raise ValueError(
                f"source {source.name} expected {suffix} input, got file {input_path.name}"
            )
        items = [SourceInputFile(path=input_path, rel_no_suffix=Path(input_path.stem))]
        report = _base_report(len(items))
        items = _apply_path_excludes(items, report)
        return items, input_path.parent, report

    if not input_path.is_dir():
        raise FileNotFoundError(f"source input is neither file nor directory: {input_path}")

    files = sorted(
        path for path in input_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() == suffix
    )
    if not files:
        raise ValueError(f"no {suffix} files found for source {source.name}: {input_path}")

    items = [
        SourceInputFile(path=path, rel_no_suffix=path.relative_to(input_path).with_suffix(""))
        for path in files
    ]

    report = _base_report(len(items))
    items = _apply_path_excludes(items, report)

    # Apply metadata filtering for seed_csv sources
    if source.type == "seed_csv" and source.metadata_csv is not None:
        items, report = _filter_seed_csv_by_metadata(
            source,
            items,
            input_path,
            quiet=quiet,
            report=report,
        )
        if not items:
            raise ValueError(
                f"no files remain after metadata filtering for source {source.name}: {input_path}"
            )

    return items, input_path, report


def _collect_source_files(source: DatasetSourceSpec) -> tuple[list[SourceInputFile], Path]:
    items, input_path, _report = _collect_source_files_with_report(source, quiet=False)
    return items, input_path


def build_source_conversion_tasks(
    source: DatasetSourceSpec,
    out_dir: Path,
    *,
    preprocess: DatasetPreprocessSpec | None = None,
) -> list[ConversionTask]:
    items, _scan_root = _collect_source_files(source)
    output_map: dict[Path, SourceInputFile] = {}
    xml_path = None
    if source.type == "bvh":
        xml_path = str(_resolve_bvh_xml_path(source))
    preprocess_spec = DatasetPreprocessSpec() if preprocess is None else preprocess

    tasks: list[ConversionTask] = []
    for item in items:
        out_npz = out_dir / item.rel_no_suffix.with_suffix(".npz")
        if out_npz in output_map:
            prev = output_map[out_npz]
            raise ValueError(
                f"output collision detected for source {source.name}: "
                f"{prev.path} and {item.path} -> {out_npz}"
            )
        output_map[out_npz] = item
        tasks.append(
            ConversionTask(
                source_name=source.name,
                source_type=source.type,
                input_path=str(item.path),
                output_path=str(out_npz),
                bvh_format=source.bvh_format,
                robot_name=source.robot_name,
                max_frames=source.max_frames,
                mocap_xml=xml_path,
                preprocess=preprocess_spec,
            )
        )
    return tasks


def _filtered_marker_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}{_FILTERED_MARKER_SUFFIX}")


def _conversion_task_signature(task: ConversionTask) -> dict[str, Any]:
    input_path = Path(task.input_path)
    stat = input_path.stat()
    signature = {
        "source_name": task.source_name,
        "source_type": task.source_type,
        "input_path": str(input_path),
        "input_size": int(stat.st_size),
        "input_mtime_ns": int(stat.st_mtime_ns),
        "bvh_format": task.bvh_format,
        "robot_name": task.robot_name,
        "max_frames": int(task.max_frames),
        "mocap_xml": task.mocap_xml,
        "preprocess": task.preprocess.to_dict(),
    }
    return json.loads(json.dumps(signature, sort_keys=True))


def _filtered_marker_matches(task: ConversionTask) -> bool:
    marker_path = _filtered_marker_path(Path(task.output_path))
    if not marker_path.is_file():
        return False
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        marker_path.unlink(missing_ok=True)
        return False
    if payload.get("signature") == _conversion_task_signature(task):
        return True
    marker_path.unlink(missing_ok=True)
    return False


def _write_filtered_marker(task: ConversionTask, reason: str) -> None:
    marker_path = _filtered_marker_path(Path(task.output_path))
    write_json(
        marker_path,
        {
            "filtered": True,
            "reason": reason,
            "signature": _conversion_task_signature(task),
        },
    )


def _clear_filtered_marker(output_path: Path) -> None:
    _filtered_marker_path(output_path).unlink(missing_ok=True)


def _prune_unexpected_source_outputs(out_dir: Path, tasks: list[ConversionTask]) -> None:
    if not out_dir.is_dir():
        return
    expected = {Path(task.output_path).resolve() for task in tasks}
    for npz_path in sorted(out_dir.rglob("*.npz")):
        if npz_path.resolve() not in expected:
            npz_path.unlink()
            _clear_filtered_marker(npz_path)
    for marker_path in sorted(out_dir.rglob(f"*.npz{_FILTERED_MARKER_SUFFIX}")):
        output_path = Path(str(marker_path)[: -len(_FILTERED_MARKER_SUFFIX)])
        if output_path.resolve() not in expected:
            marker_path.unlink(missing_ok=True)


def _current_source_npz_files(source: DatasetSourceSpec, source_dir: Path) -> list[Path]:
    items, _scan_root, _report = _collect_source_files_with_report(source, quiet=True)
    allowed_rels = {item.rel_no_suffix.as_posix() for item in items}
    return [
        npz_path
        for npz_path in sorted(source_dir.rglob("*.npz"))
        if npz_path.relative_to(source_dir).with_suffix("").as_posix() in allowed_rels
    ]


def _pending_tasks(tasks: list[ConversionTask]) -> list[ConversionTask]:
    pending: list[ConversionTask] = []
    for task in tasks:
        output_path = Path(task.output_path)
        if output_path.is_file():
            _clear_filtered_marker(output_path)
            continue
        if _filtered_marker_matches(task):
            continue
        pending.append(task)
    return pending


def _build_source_filter_reports(spec: DatasetSpec) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for source in spec.sources:
        _, _, report = _collect_source_files_with_report(source, quiet=True)
        reports.append(report)
    return reports


def _get_fk_extractor() -> MotionFkExtractor:
    global _PROCESS_FK_EXTRACTOR
    if _PROCESS_FK_EXTRACTOR is None:
        _PROCESS_FK_EXTRACTOR = MotionFkExtractor()
    return _PROCESS_FK_EXTRACTOR


def _get_bvh_model(xml_path: str) -> mujoco.MjModel:
    model = _PROCESS_BVH_MODEL_CACHE.get(xml_path)
    if model is None:
        model = mujoco.MjModel.from_xml_path(xml_path)
        _PROCESS_BVH_MODEL_CACHE[xml_path] = model
    return model


def _maybe_preprocess_clip_dict(
    clip_dict: dict[str, Any],
    *,
    preprocess: DatasetPreprocessSpec,
    clip_label: str,
) -> dict[str, Any]:
    processed = preprocess_clip_dict(clip_dict, spec=preprocess, clip_label=clip_label)
    inspect_clip_dict(processed)
    return processed


def _maybe_preprocess_npz_file(
    npz_path: Path,
    *,
    preprocess: DatasetPreprocessSpec,
    clip_label: str,
) -> None:
    payload = np.load(npz_path, allow_pickle=True)
    clip_dict = {key: payload[key] for key in payload.files}
    processed = _maybe_preprocess_clip_dict(
        clip_dict,
        preprocess=preprocess,
        clip_label=clip_label,
    )
    np.savez(npz_path, **processed)


def _convert_task(task: ConversionTask) -> str | FilteredClipResult:
    input_path = Path(task.input_path)
    output_path = Path(task.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if task.source_type == "npz":
            payload = np.load(input_path, allow_pickle=True)
            clip_dict = {key: payload[key] for key in payload.files}
            clip_label = f"{task.source_name}:{input_path.name}"
            processed = _maybe_preprocess_clip_dict(
                clip_dict,
                preprocess=task.preprocess,
                clip_label=clip_label,
            )
            np.savez(output_path, **processed)
            inspect_npz(output_path)
            _clear_filtered_marker(output_path)
            return str(output_path)

        extractor = _get_fk_extractor()
        if task.source_type == "pkl":
            convert_pkl_to_npz(str(input_path), str(output_path), extractor=extractor)
            clip_label = f"{task.source_name}:{input_path.name}"
            _maybe_preprocess_npz_file(
                output_path,
                preprocess=task.preprocess,
                clip_label=clip_label,
            )
            inspect_npz(output_path)
            _clear_filtered_marker(output_path)
            return str(output_path)

        if task.source_type == "seed_csv":
            arrays = convert_seed_csv_to_arrays(str(input_path), extractor=extractor)
            clip_label = f"{task.source_name}:{input_path.name}"
            arrays = _maybe_preprocess_clip_dict(
                arrays,
                preprocess=task.preprocess,
                clip_label=clip_label,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(output_path), **arrays)
            inspect_npz(output_path)
            _clear_filtered_marker(output_path)
            return str(output_path)

        if task.source_type == "bvh":
            if task.bvh_format is None or task.mocap_xml is None:
                raise ValueError("bvh conversion task missing bvh_format or mocap_xml")
            model = _get_bvh_model(task.mocap_xml)
            with TemporaryDirectory(prefix="teleopit_dataset_bvh_") as tmp_dir:
                tmp_pkl = Path(tmp_dir) / f"{input_path.stem}.pkl"
                convert_bvh_to_retarget_pkl(
                    input_path,
                    tmp_pkl,
                    task.bvh_format,
                    task.robot_name,
                    task.max_frames,
                    model,
                )
                convert_pkl_to_npz(str(tmp_pkl), str(output_path), extractor=extractor)
            clip_label = f"{task.source_name}:{input_path.name}"
            _maybe_preprocess_npz_file(
                output_path,
                preprocess=task.preprocess,
                clip_label=clip_label,
            )
            inspect_npz(output_path)
            _clear_filtered_marker(output_path)
            return str(output_path)

        raise ValueError(f"unsupported source type: {task.source_type}")
    except ValueError as exc:
        clip_label = f"{task.source_name}:{input_path.name}"
        if str(exc).startswith(f"{clip_label}:"):
            output_path.unlink(missing_ok=True)
            _write_filtered_marker(task, str(exc))
            return FilteredClipResult(input_path=str(input_path), reason=str(exc))
        raise RuntimeError(
            f"failed converting source={task.source_name} input={input_path}: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"failed converting source={task.source_name} input={input_path}: {exc}"
        ) from exc


def _run_conversion_tasks_serial(tasks: list[ConversionTask]) -> None:
    total = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        result = _convert_task(task)
        if isinstance(result, FilteredClipResult):
            print(f"[FILTER] {result.reason}")
            continue
        print(f"[CONVERT] {idx}/{total} source={task.source_name} -> {_display_path(Path(task.output_path))}")


def run_conversion_tasks(tasks: list[ConversionTask], *, jobs: int = DEFAULT_JOBS) -> None:
    if jobs <= 0:
        raise ValueError(f"jobs must be > 0, got {jobs}")
    if not tasks:
        return

    total = len(tasks)
    if jobs == 1 or total == 1:
        _run_conversion_tasks_serial(tasks)
        return

    max_workers = min(jobs, total)
    ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            future_map = {executor.submit(_convert_task, task): task for task in tasks}
            completed = 0
            try:
                for future in as_completed(future_map):
                    task = future_map[future]
                    result = future.result()
                    completed += 1
                    if isinstance(result, FilteredClipResult):
                        print(f"[FILTER] {result.reason}")
                    else:
                        print(
                            f"[CONVERT] {completed}/{total} "
                            f"source={task.source_name} -> {_display_path(Path(task.output_path))}"
                        )
            except Exception:
                for future in future_map:
                    future.cancel()
                raise
    except (PermissionError, OSError) as exc:
        print(f"[WARN] process pool unavailable ({exc}); falling back to serial conversion")
        _run_conversion_tasks_serial(tasks)


def convert_source_to_npz_clips(
    source: DatasetSourceSpec,
    output_dir: Path,
    *,
    force: bool = False,
    jobs: int = DEFAULT_JOBS,
    preprocess: DatasetPreprocessSpec | None = None,
) -> dict[str, Any]:
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    tasks = build_source_conversion_tasks(source, output_dir, preprocess=preprocess)
    _prune_unexpected_source_outputs(output_dir, tasks)
    pending = _pending_tasks(tasks)
    if tasks and not pending:
        print(f"[CACHE] reusing source={source.name} clips: {_display_path(output_dir)}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_conversion_tasks(pending, jobs=jobs)

    npz_files = _current_source_npz_files(source, output_dir)
    if not npz_files:
        raise ValueError(f"no converted npz clips found for source {source.name}: {output_dir}")
    return {
        "source": source.name,
        "type": source.type,
        "output_dir": str(output_dir),
        "clips": len(npz_files),
    }


def convert_sources_to_npz(
    spec: DatasetSpec,
    *,
    paths: DatasetPaths,
    force: bool = False,
    jobs: int = DEFAULT_JOBS,
) -> dict[str, Path]:
    if jobs <= 0:
        raise ValueError(f"jobs must be > 0, got {jobs}")
    _ensure_source_inputs_do_not_overlap_intermediate_clips(spec, paths.clips_root)
    if force and paths.dataset_dir.exists():
        shutil.rmtree(paths.dataset_dir)
    paths.clips_root.mkdir(parents=True, exist_ok=True)

    source_out_dirs = {source.name: paths.clips_root / source.name for source in spec.sources}
    pending_tasks: list[ConversionTask] = []
    for source in spec.sources:
        out_dir = source_out_dirs[source.name]
        tasks = build_source_conversion_tasks(source, out_dir, preprocess=spec.preprocess)
        _prune_unexpected_source_outputs(out_dir, tasks)
        pending = _pending_tasks(tasks)
        if tasks and not pending:
            print(f"[CACHE] reusing source={source.name} clips: {_display_path(out_dir)}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        pending_tasks.extend(pending)

    run_conversion_tasks(pending_tasks, jobs=jobs)
    return source_out_dirs


def collect_clip_rows(spec: DatasetSpec, *, paths: DatasetPaths) -> list[DatasetClipRow]:
    rows: list[DatasetClipRow] = []
    for source in spec.sources:
        source_dir = paths.clips_root / source.name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"expected converted npz dir for {source.name}: {source_dir}")
        npz_files = _current_source_npz_files(source, source_dir)
        if not npz_files:
            raise ValueError(f"no converted npz clips found for source {source.name}: {source_dir}")
        for npz_path in npz_files:
            meta = inspect_npz(npz_path)
            rel_under_source = npz_path.relative_to(source_dir).with_suffix("").as_posix()
            rows.append(
                DatasetClipRow(
                    clip_id=f"{source.name}:{rel_under_source}",
                    source=source.name,
                    file_rel=_display_path(npz_path),
                    num_frames=meta.num_frames,
                    fps=meta.fps,
                    resolved_npz_path=str(npz_path),
                )
            )
    if not rows:
        raise ValueError("no clip rows collected")
    return rows


def run_sample_fk_checks(
    rows: list[DatasetClipRow],
    *,
    sample_clips_per_source: int = DEFAULT_FK_SAMPLE_CLIPS,
    sample_frames: int = DEFAULT_FK_SAMPLE_FRAMES,
) -> list[FkCheckSummary]:
    summaries: list[FkCheckSummary] = []
    by_source: dict[str, list[DatasetClipRow]] = {}
    for row in rows:
        by_source.setdefault(row.source, []).append(row)

    for source, source_rows in sorted(by_source.items()):
        for row in sorted(source_rows, key=lambda item: item.clip_id)[:sample_clips_per_source]:
            stats = compute_npz_fk_consistency(row.resolved_npz_path, sample_count=sample_frames)
            if stats.pos_max > 1e-3 or stats.quat_mean > 0.05 or stats.quat_p95 > 0.10:
                raise ValueError(
                    f"FK consistency check failed for {row.clip_id}: "
                    f"pos_max={stats.pos_max:.6e}, quat_mean={stats.quat_mean:.6e}, "
                    f"quat_p95={stats.quat_p95:.6e}"
                )
            summaries.append(
                FkCheckSummary(
                    source=source,
                    clip=row.clip_id,
                    pos_max=stats.pos_max,
                    quat_mean=stats.quat_mean,
                    quat_p95=stats.quat_p95,
                )
            )
    return summaries


_ARRAY_KEYS = FULL_CLIP_ARRAY_KEYS


def _batch_convert_chunk(
    file_paths: list[str],
    target_fps: int,
    output_path: str,
    label: str,
    preprocess: DatasetPreprocessSpec,
) -> dict[str, Any]:
    """Worker: convert a batch of PKL/seed_csv files and write one HDF5 shard.

    Designed to run in a spawned subprocess via ProcessPoolExecutor.
    """
    from train_mimic.data.motion_fk import MotionFkExtractor
    from train_mimic.scripts.convert_pkl_to_npz import convert_pkl_to_arrays, convert_seed_csv_to_arrays

    extractor = MotionFkExtractor()
    acc: dict[str, list[np.ndarray]] = {k: [] for k in _ARRAY_KEYS}
    clip_lengths: list[int] = []
    body_names: np.ndarray | None = None
    total = len(file_paths)
    filtered = 0
    kept_file_paths: list[str] = []

    for i, file_path in enumerate(file_paths):
        try:
            if file_path.endswith(".csv"):
                arrays = convert_seed_csv_to_arrays(file_path, extractor=extractor)
            else:
                arrays = convert_pkl_to_arrays(file_path, extractor=extractor)
        except Exception as exc:
            raise RuntimeError(f"failed converting {file_path}: {exc}") from exc

        fps = int(arrays.pop("fps"))
        cur_body_names = np.asarray(arrays.pop("body_names"))
        if body_names is None:
            body_names = cur_body_names
        elif not np.array_equal(cur_body_names, body_names):
            raise ValueError(f"inconsistent body_names while batch-converting {file_path}")

        # Resample if source fps differs from target
        if fps != target_fps:
            old_t = arrays["joint_pos"].shape[0]
            new_t = max(1, round(old_t * target_fps / fps))
            for key in _ARRAY_KEYS:
                arrays[key] = resample_along_time(arrays[key], new_t)
            qn_root = np.linalg.norm(arrays["root_quat_w"], axis=-1, keepdims=True)
            arrays["root_quat_w"] = arrays["root_quat_w"] / np.where(qn_root < 1e-8, 1.0, qn_root)
            qn = np.linalg.norm(arrays["body_quat_w"], axis=-1, keepdims=True)
            arrays["body_quat_w"] = arrays["body_quat_w"] / np.where(qn < 1e-8, 1.0, qn)

        clip_dict = {"fps": fps, **arrays, "body_names": cur_body_names}
        if fps != target_fps:
            clip_dict["fps"] = target_fps
        clip_label = f"{label}:{Path(file_path).name}"
        try:
            clip_dict = _maybe_preprocess_clip_dict(
                clip_dict,
                preprocess=preprocess,
                clip_label=clip_label,
            )
        except ValueError as exc:
            if str(exc).startswith(f"{clip_label}:"):
                filtered += 1
                print(f"[FILTER] {exc}", flush=True)
                continue
            raise

        for key in _ARRAY_KEYS:
            acc[key].append(np.asarray(clip_dict[key]))
        clip_lengths.append(int(np.asarray(clip_dict["joint_pos"]).shape[0]))
        kept_file_paths.append(file_path)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"[BATCH] {label}: {i + 1}/{total}", flush=True)

    kept = len(clip_lengths)
    if kept == 0:
        print(f"[BATCH] {label}: all {total} clips filtered out", flush=True)
        return {
            "output": output_path,
            "clips": 0,
            "num_clips": 0,
            "frames": 0,
            "fps": target_fps,
            "duration_s": 0.0,
            "clip_lengths": [],
            "source_clip_lengths": [],
            "kept_file_paths": [],
        }

    # Build merged chunk
    clip_lengths_arr = np.array(clip_lengths, dtype=np.int64)
    clip_starts = np.zeros(kept, dtype=np.int64)
    if kept > 1:
        clip_starts[1:] = np.cumsum(clip_lengths_arr[:-1])

    merged = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    merged["fps"] = target_fps
    merged["body_names"] = body_names
    merged["clip_starts"] = clip_starts
    merged["clip_lengths"] = clip_lengths_arr
    merged["clip_fps"] = np.full(kept, target_fps, dtype=np.int64)

    shard_info = write_hdf5_motion_shard(
        merged,
        Path(output_path),
        max_window_frames=DEFAULT_HDF5_MAX_WINDOW_FRAMES,
        overlap_frames=DEFAULT_HDF5_WINDOW_OVERLAP_FRAMES,
    )

    total_frames = int(merged["joint_pos"].shape[0])
    print(
        f"[BATCH] {label}: done, {kept} kept / {total} total clips, "
        f"{filtered} filtered, {total_frames} frames -> {Path(output_path).name}",
        flush=True,
    )
    hdf5_windows = int(shard_info["clips"])
    return {
        "output": output_path,
        "clips": hdf5_windows,
        "num_clips": hdf5_windows,
        "source_clips": kept,
        "frames": total_frames,
        "fps": target_fps,
        "duration_s": total_frames / max(target_fps, 1),
        "clip_lengths": list(shard_info["clip_lengths"]),
        "source_clip_lengths": clip_lengths,
        "kept_file_paths": kept_file_paths,
    }


def _shard_stats(
    *,
    output_dir: Path,
    shard_infos: list[dict[str, Any]],
    fps: int,
) -> dict[str, Any]:
    total_clips = sum(len(info["clip_lengths"]) for info in shard_infos)
    total_frames = sum(int(info.get("frames", 0)) for info in shard_infos)
    if total_frames <= 0:
        total_frames = sum(
            sum(int(length) for length in info.get("source_clip_lengths", info["clip_lengths"]))
            for info in shard_infos
        )
    return {
        "output": str(output_dir),
        "shards": len(shard_infos),
        "clips": total_clips,
        "num_clips": total_clips,
        "frames": total_frames,
        "fps": fps,
        "duration_s": float(total_frames / max(fps, 1)),
    }


def _batch_convert_split(
    file_paths: list[str],
    target_fps: int,
    output_dir: Path,
    jobs: int,
    label: str,
    preprocess: DatasetPreprocessSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert clips for one dataset using parallel chunk workers."""
    if not file_paths:
        raise ValueError(f"no clips for dataset {label}")

    num_workers = min(jobs, len(file_paths))
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_workers <= 1:
        shard_path = shard_output_path(output_dir, 0)
        stats = _batch_convert_chunk(
            file_paths, target_fps, str(shard_path), label, preprocess,
        )
        if int(stats["clips"]) <= 0:
            raise ValueError(f"no valid clips remain for dataset {label} after preprocessing")
        shard_infos = [{
            "path": shard_path,
            "clips": int(stats.get("clips", 0)),
            "frames": int(stats.get("frames", 0)),
            "clip_lengths": list(stats.pop("clip_lengths", [])),
            "source_clip_lengths": list(stats.pop("source_clip_lengths", [])),
            "kept_file_paths": list(stats.pop("kept_file_paths", [])),
        }]
        return _shard_stats(output_dir=output_dir, shard_infos=shard_infos, fps=target_fps), shard_infos

    # Split into chunks, one per worker
    chunk_size = (len(file_paths) + num_workers - 1) // num_workers
    chunk_args: list[tuple[list[str], int, str, str, DatasetPreprocessSpec]] = []
    for i in range(num_workers):
        start = i * chunk_size
        end = min(start + chunk_size, len(file_paths))
        if start >= len(file_paths):
            break
        chunk_out = str(output_dir / f".{label}_chunk_{i}.h5")
        chunk_args.append((
            file_paths[start:end],
            target_fps,
            chunk_out,
            f"{label}[{i}]",
            preprocess,
        ))

    chunk_results: dict[str, dict[str, Any]] = {}
    ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=len(chunk_args), mp_context=ctx) as executor:
            futures = {
                executor.submit(_batch_convert_chunk, *args): args[2]
                for args in chunk_args
            }
            try:
                for future in as_completed(futures):
                    result = future.result()
                    chunk_results[str(result["output"])] = result
            except Exception:
                for future in futures:
                    future.cancel()
                raise
    except (PermissionError, OSError):
        print(f"[WARN] process pool unavailable; falling back to serial for {label}")
        shard_path = shard_output_path(output_dir, 0)
        stats = _batch_convert_chunk(
            file_paths,
            target_fps,
            str(shard_path),
            label,
            preprocess,
        )
        if int(stats["clips"]) <= 0:
            raise ValueError(f"no valid clips remain for dataset {label} after preprocessing")
        shard_infos = [{
            "path": shard_path,
            "clips": int(stats.get("clips", 0)),
            "frames": int(stats.get("frames", 0)),
            "clip_lengths": list(stats.pop("clip_lengths", [])),
            "source_clip_lengths": list(stats.pop("source_clip_lengths", [])),
            "kept_file_paths": list(stats.pop("kept_file_paths", [])),
        }]
        return _shard_stats(output_dir=output_dir, shard_infos=shard_infos, fps=target_fps), shard_infos

    shard_infos: list[dict[str, Any]] = []
    shard_index = 0
    for args in chunk_args:
        tmp_path = Path(args[2])
        chunk_stat = chunk_results.get(args[2], {})
        if int(chunk_stat.get("clips", 0)) <= 0:
            tmp_path.unlink(missing_ok=True)
            continue
        final_path = shard_output_path(output_dir, shard_index)
        shard_index += 1
        tmp_path.replace(final_path)
        shard_infos.append({
            "path": final_path,
            "clips": int(chunk_stat.get("clips", 0)),
            "frames": int(chunk_stat.get("frames", 0)),
            "clip_lengths": list(chunk_stat.get("clip_lengths", [])),
            "source_clip_lengths": list(chunk_stat.get("source_clip_lengths", [])),
            "kept_file_paths": list(chunk_stat.get("kept_file_paths", [])),
        })

    if not shard_infos:
        raise ValueError(f"no valid clips remain for dataset {label} after preprocessing")

    stats = _shard_stats(output_dir=output_dir, shard_infos=shard_infos, fps=target_fps)
    print(
        f"[SHARDS] {label}: {stats['shards']} shards, "
        f"{stats['clips']} clips, {stats['frames']} frames ({stats['duration_s']:.1f}s)",
        flush=True,
    )
    return stats, shard_infos


def _build_dataset_batch(
    spec: DatasetSpec,
    *,
    paths: DatasetPaths,
    force: bool = False,
    skip_fk_check: bool = False,
    skip_validate: bool = False,
    jobs: int = DEFAULT_JOBS,
) -> dict[str, Any]:
    """Batch build: enumerate -> parallel chunk convert -> minimal HDF5 shards.

    Skips writing individual clip files. Each worker converts a batch of PKL or
    seed CSV files in-memory and writes one minimal HDF5 shard.
    """
    if force and paths.dataset_dir.exists():
        shutil.rmtree(paths.dataset_dir)

    clip_entries: list[tuple[str, str, str]] = []
    source_filter_reports: list[dict[str, Any]] = []
    for source in spec.sources:
        items, _, filter_report = _collect_source_files_with_report(source, quiet=False)
        source_filter_reports.append(filter_report)
        for item in items:
            clip_id = f"{source.name}:{item.rel_no_suffix.as_posix()}"
            clip_entries.append((str(item.path), clip_id, source.name))

    if not clip_entries:
        raise ValueError("dataset must contain at least 1 clip")

    print(
        f"[DATASET] {spec.name}: {len(clip_entries)} clips, jobs={jobs}",
        flush=True,
    )

    _clear_existing_motion_shards(paths.dataset_dir)
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    stats, shard_infos = _batch_convert_split(
        [e[0] for e in clip_entries],
        spec.target_fps,
        paths.dataset_dir,
        jobs,
        spec.name,
        spec.preprocess,
    )

    return {
        "dataset": spec.name,
        "target_fps": spec.target_fps,
        "dataset_dir": str(paths.dataset_dir),
        "skip_validate": bool(skip_validate),
        "skip_fk_check": bool(skip_fk_check),
        "jobs": int(jobs),
        "preprocess": spec.preprocess.to_dict(),
        "source_filters": source_filter_reports,
        "stats": stats,
        "shards": [str(info["path"]) for info in shard_infos],
        "input_clips": len(clip_entries),
        "fk_checks": [],
    }


def build_dataset_from_spec(
    spec: DatasetSpec,
    *,
    force: bool = False,
    skip_fk_check: bool = False,
    skip_validate: bool = False,
    jobs: int = DEFAULT_JOBS,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    paths = resolve_dataset_paths(spec, output_root=output_root)

    # Use batch mode for pkl/seed_csv-only datasets (no intermediate clip files)
    all_batch_eligible = all(source.type in ("pkl", "seed_csv") for source in spec.sources)
    if all_batch_eligible:
        return _build_dataset_batch(
            spec,
            paths=paths,
            force=force,
            skip_fk_check=skip_fk_check,
            skip_validate=skip_validate,
            jobs=jobs,
        )

    # Per-file mode for BVH/NPZ sources. Converted clips are temporary build
    # inputs and are rebuilt every time; final training data is the minimal
    # shard(s) in dataset_dir.
    _ensure_source_inputs_do_not_overlap_intermediate_clips(spec, paths.clips_root)
    _clear_intermediate_clips(paths.clips_root)
    convert_sources_to_npz(spec, paths=paths, force=force, jobs=jobs)
    rows = collect_clip_rows(spec, paths=paths)

    fk_checks: list[FkCheckSummary] = []
    if not skip_fk_check:
        fk_checks = run_sample_fk_checks(rows)

    _clear_existing_motion_shards(paths.dataset_dir)
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_output_path(paths.dataset_dir, 0)
    tmp_npz = paths.dataset_dir / ".merged_tmp.npz"
    stats = merge_npz_files(
        [Path(row.resolved_npz_path) for row in rows],
        tmp_npz,
        target_fps=spec.target_fps,
    )
    payload_npz = np.load(tmp_npz, allow_pickle=True)
    payload = {key: payload_npz[key] for key in payload_npz.files}
    shard_info = write_hdf5_motion_shard(payload, shard_path)
    tmp_npz.unlink(missing_ok=True)
    _clear_intermediate_clips(paths.clips_root)

    stats["output"] = str(paths.dataset_dir)
    stats["shards"] = 1
    stats["clips"] = int(shard_info["clips"])
    stats["num_clips"] = int(shard_info["clips"])

    return {
        "dataset": spec.name,
        "target_fps": spec.target_fps,
        "dataset_dir": str(paths.dataset_dir),
        "clips_dir": str(paths.clips_root),
        "skip_validate": bool(skip_validate),
        "skip_fk_check": bool(skip_fk_check),
        "jobs": int(jobs),
        "preprocess": spec.preprocess.to_dict(),
        "stats": stats,
        "shards": [str(shard_path)],
        "input_clips": len(rows),
        "fk_checks": [item.to_dict() for item in fk_checks],
    }
