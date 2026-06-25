from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest
import h5py

from train_mimic.data import dataset_builder
from train_mimic.data.dataset_lib import compute_dataset_stats, write_hdf5_motion_shard
from train_mimic.data.dataset_builder import (
    DatasetClipRow,
    SourceInputFile,
    DatasetSourceSpec,
    DatasetSpec,
    build_dataset_from_spec,
    convert_source_to_npz_clips,
    load_dataset_spec,
)
from train_mimic.data.dataset_lib import inspect_npz, merge_clip_dicts
from train_mimic.data.motion_fk import (
    DEFAULT_G1_XML_PATH,
    MotionFkExtractor,
    normalize_quaternion,
    quat_rotate_inverse,
    quat_wxyz_to_xyzw,
)
from train_mimic.scripts.convert_pkl_to_npz import _MJLAB_G1_BODY_NAMES, convert_pkl_to_npz

pytestmark = [
    pytest.mark.skipif(
        not DEFAULT_G1_XML_PATH.is_file(),
        reason="GMR G1 assets not downloaded",
    )
]


def _synthetic_motion_payload() -> dict[str, object]:
    extractor = MotionFkExtractor()
    body_names = list(_MJLAB_G1_BODY_NAMES)

    root_pos = np.asarray(
        [
            [0.0, 0.0, 0.76],
            [0.03, -0.01, 0.77],
            [0.07, -0.02, 0.775],
            [0.10, -0.015, 0.78],
        ],
        dtype=np.float32,
    )
    root_quat_wxyz = normalize_quaternion(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9950042, 0.0, 0.0, 0.0998334],
                [0.9800666, 0.0, 0.0, 0.1986693],
                [0.9553365, 0.0, 0.0, 0.2955202],
            ],
            dtype=np.float32,
        )
    )
    dof_pos = np.zeros((4, extractor.num_actions), dtype=np.float32)
    dof_pos[:, 3] = np.asarray([0.05, 0.15, 0.30, 0.20], dtype=np.float32)
    dof_pos[:, 9] = np.asarray([-0.05, -0.10, -0.20, -0.12], dtype=np.float32)

    body_pos_w, _ = extractor.extract(root_pos, root_quat_wxyz, dof_pos, body_names)
    local_body_pos = quat_rotate_inverse(
        root_quat_wxyz[:, None, :],
        body_pos_w - root_pos[:, None, :],
    ).astype(np.float32)

    return {
        "fps": 30,
        "root_pos": root_pos,
        "root_rot": quat_wxyz_to_xyzw(root_quat_wxyz).astype(np.float32),
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": body_names,
    }


def _write_pkl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(_synthetic_motion_payload(), handle)


def _write_npz_from_pkl(path: Path) -> None:
    pkl_path = path.with_suffix(".pkl")
    _write_pkl(pkl_path)
    convert_pkl_to_npz(str(pkl_path), str(path))


def test_load_dataset_spec_parses_typed_sources(tmp_path: Path) -> None:
    spec_path = tmp_path / "demo.yaml"
    spec_path.write_text(
        f"""name: demo
target_fps: 30
sources:
  - name: clips
    type: npz
    input: {tmp_path / 'npz_source'}
  - name: lafan1
    type: bvh
    input: {tmp_path / 'lafan1'}
    bvh_format: lafan1
    robot_name: unitree_g1
""",
        encoding="utf-8",
    )

    spec = load_dataset_spec(spec_path)
    assert spec.name == "demo"
    assert spec.target_fps == 30
    assert spec.sources[0].type == "npz"
    assert spec.sources[1].type == "bvh"
    assert spec.sources[1].bvh_format == "lafan1"


def test_load_dataset_spec_parses_preprocess(tmp_path: Path) -> None:
    spec_path = tmp_path / "preprocessed.yaml"
    spec_path.write_text(
        f"""name: demo
target_fps: 30
preprocess:
  normalize_root_xy: true
  ground_align: none
  min_frames: 10
  max_all_off_ground_s: 0.5
  off_ground_height: 0.08
sources:
  - name: clips
    type: npz
    input: {tmp_path / 'npz_source'}
    exclude_patterns: ["*obstacle*"]
""",
        encoding="utf-8",
    )

    spec = load_dataset_spec(spec_path)
    assert spec.preprocess.normalize_root_xy is True
    assert spec.preprocess.ground_align == "none"
    assert spec.preprocess.min_frames == 10
    assert spec.preprocess.max_all_off_ground_s == 0.5
    assert spec.preprocess.off_ground_height == 0.08
    assert spec.sources[0].exclude_patterns == ("*obstacle*",)


def test_load_dataset_spec_parses_seed_filter_preset(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "seed_metadata.csv"
    metadata_csv.write_text("move_g1_path,is_mirror\n", encoding="utf-8")
    spec_path = tmp_path / "seed.yaml"
    spec_path.write_text(
        f"""name: seed_demo
target_fps: 30
sources:
  - name: seed
    type: seed_csv
    input: {tmp_path / 'seed_source'}
    metadata_csv: {metadata_csv}
    seed_filter_preset: groot_strict
    filters:
      is_mirror: [false]
""",
        encoding="utf-8",
    )

    spec = load_dataset_spec(spec_path)
    assert spec.sources[0].seed_filter_preset == "groot_strict"


def test_load_dataset_spec_rejects_seed_filter_preset_on_non_seed_source(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "seed_metadata.csv"
    metadata_csv.write_text("move_g1_path,is_mirror\n", encoding="utf-8")
    spec_path = tmp_path / "bad_seed_preset.yaml"
    spec_path.write_text(
        f"""name: demo
target_fps: 30
sources:
  - name: clips
    type: npz
    input: {tmp_path / 'npz_source'}
    metadata_csv: {metadata_csv}
    seed_filter_preset: groot_strict
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="supported only for seed_csv sources"):
        load_dataset_spec(spec_path)


def test_load_dataset_spec_rejects_unknown_seed_filter_preset(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "seed_metadata.csv"
    metadata_csv.write_text("move_g1_path,is_mirror\n", encoding="utf-8")
    spec_path = tmp_path / "bad_seed_preset.yaml"
    spec_path.write_text(
        f"""name: demo
target_fps: 30
sources:
  - name: seed
    type: seed_csv
    input: {tmp_path / 'seed_source'}
    metadata_csv: {metadata_csv}
    seed_filter_preset: unknown_preset
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown seed_filter_preset"):
        load_dataset_spec(spec_path)


def test_load_dataset_spec_rejects_bvh_without_format(tmp_path: Path) -> None:
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text(
        """name: demo
target_fps: 30
sources:
  - name: broken
    type: bvh
    input: data/lafan1_bvh
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires bvh_format"):
        load_dataset_spec(spec_path)


def test_convert_source_to_npz_clips_handles_pkl_source(tmp_path: Path) -> None:
    input_dir = tmp_path / "pkl_source"
    _write_pkl(input_dir / "clip.pkl")

    source = DatasetSourceSpec(name="pkl_src", type="pkl", input=str(input_dir))
    output_dir = tmp_path / "dataset" / "clips" / "pkl_src"
    report = convert_source_to_npz_clips(source, output_dir, jobs=1)

    assert report["clips"] == 1
    clip_path = output_dir / "clip.npz"
    assert clip_path.is_file()
    assert inspect_npz(clip_path).fps == 30


def test_convert_source_to_npz_clips_handles_bvh_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "bvh_source"
    input_dir.mkdir()
    (input_dir / "clip.bvh").write_text("HIERARCHY\n", encoding="utf-8")
    fake_xml = tmp_path / "fake.xml"
    fake_xml.write_text("<xml />", encoding="utf-8")
    payload = _synthetic_motion_payload()

    def fake_convert_bvh_to_retarget_pkl(
        bvh_path: Path,
        output_pkl: Path,
        bvh_format: str,
        robot_name: str,
        max_frames: int,
        model: object,
        ) -> None:
            assert bvh_path.name == "clip.bvh"
            assert bvh_format == "lafan1"
            output_pkl.parent.mkdir(parents=True, exist_ok=True)
            with output_pkl.open("wb") as handle:
                pickle.dump(payload, handle)

    monkeypatch.setattr(
        "train_mimic.data.dataset_builder.mocap_xml_path",
        lambda _, robot_name="unitree_g1": fake_xml if robot_name == "unitree_g1" else None,
    )
    monkeypatch.setattr("train_mimic.data.dataset_builder.convert_bvh_to_retarget_pkl", fake_convert_bvh_to_retarget_pkl)
    monkeypatch.setattr(
        "train_mimic.data.dataset_builder.mujoco.MjModel.from_xml_path",
        lambda _: object(),
    )

    source = DatasetSourceSpec(
        name="bvh_src",
        type="bvh",
        input=str(input_dir),
        bvh_format="lafan1",
        robot_name="unitree_g1",
    )
    output_dir = tmp_path / "dataset" / "clips" / "bvh_src"
    report = convert_source_to_npz_clips(source, output_dir, jobs=1)

    assert report["clips"] == 1
    assert (output_dir / "clip.npz").is_file()


def test_convert_source_to_npz_clips_rejects_non_g1_bvh_robot(tmp_path: Path) -> None:
    input_dir = tmp_path / "bvh_source"
    input_dir.mkdir()
    (input_dir / "clip.bvh").write_text("HIERARCHY\n", encoding="utf-8")

    source = DatasetSourceSpec(
        name="bvh_src",
        type="bvh",
        input=str(input_dir),
        bvh_format="lafan1",
        robot_name="fourier_n1",
    )

    with pytest.raises(ValueError, match="currently supports only 'unitree_g1'"):
        convert_source_to_npz_clips(source, tmp_path / "dataset" / "clips" / "bvh_src", jobs=1)


def test_convert_source_to_npz_clips_rejects_dataset_root_for_npz_source(tmp_path: Path) -> None:
    dataset_root = tmp_path / "old_dataset"
    clips_dir = dataset_root / "clips" / "src"
    clips_dir.mkdir(parents=True)
    _write_npz_from_pkl(clips_dir / "clip_a.npz")
    train_dir = dataset_root / "train"
    train_dir.mkdir(parents=True)
    _write_npz_from_pkl(train_dir / "shard_000.npz")
    (dataset_root / "build_info.json").write_text("{}", encoding="utf-8")

    source = DatasetSourceSpec(name="npz_src", type="npz", input=str(dataset_root))
    with pytest.raises(ValueError, match="points at dataset root"):
        convert_source_to_npz_clips(source, tmp_path / "dataset" / "clips" / "npz_src", jobs=1)


def test_collect_source_files_with_report_applies_seed_filter_preset(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "seed_metadata.csv"
    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "move_g1_path",
                "is_mirror",
                "content_body_position",
                "content_type_of_movement",
                "content_props",
                "filename",
                "move_name",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "move_g1_path": "g1/csv/240101/walk_forward.csv",
                "is_mirror": "False",
                "content_body_position": "standing",
                "content_type_of_movement": "walking",
                "content_props": "0",
                "filename": "walk_forward",
                "move_name": "walk_forward",
            }
        )
        writer.writerow(
            {
                "move_g1_path": "g1/csv/240101/sit_pose.csv",
                "is_mirror": "False",
                "content_body_position": "sitting",
                "content_type_of_movement": "sitting",
                "content_props": "chair",
                "filename": "sit_pose",
                "move_name": "sit_pose",
            }
        )

    source = DatasetSourceSpec(
        name="seed",
        type="seed_csv",
        input=str(tmp_path / "seed_source" / "g1" / "csv"),
        metadata_csv=str(metadata_csv),
        filters={"is_mirror": [False]},
        seed_filter_preset="groot_strict",
    )

    input_root = tmp_path / "seed_source" / "g1" / "csv"
    input_root.mkdir(parents=True, exist_ok=True)
    (input_root / "240101").mkdir(parents=True, exist_ok=True)
    (input_root / "240101" / "walk_forward.csv").write_text("placeholder", encoding="utf-8")
    (input_root / "240101" / "sit_pose.csv").write_text("placeholder", encoding="utf-8")

    items, _scan_root, report = dataset_builder._collect_source_files_with_report(source, quiet=True)

    assert [item.rel_no_suffix.as_posix() for item in items] == ["240101/walk_forward"]
    assert report["scanned_files"] == 2
    assert report["metadata_rows_matched"] == 2
    assert report["preset_rejected_rows"] == 1
    assert report["kept_files"] == 1
    assert report["filtered_files"] == 1
    assert report["preset_reject_reasons"]["content_body_position:sitting"] == 1


def test_collect_source_files_with_report_handles_single_file_source(tmp_path: Path) -> None:
    npz_path = tmp_path / "clip_a.npz"
    _write_npz_from_pkl(npz_path)
    source = DatasetSourceSpec(name="clip", type="npz", input=str(npz_path))

    items, scan_root, report = dataset_builder._collect_source_files_with_report(
        source, quiet=True
    )
    legacy_items, legacy_scan_root = dataset_builder._collect_source_files(source)

    assert scan_root == tmp_path
    assert [item.rel_no_suffix.as_posix() for item in items] == ["clip_a"]
    assert report["scanned_files"] == 1
    assert report["kept_files"] == 1
    assert legacy_scan_root == scan_root
    assert [item.rel_no_suffix.as_posix() for item in legacy_items] == ["clip_a"]


def test_collect_source_files_with_report_applies_exclude_patterns(tmp_path: Path) -> None:
    input_root = tmp_path / "lafan1"
    input_root.mkdir(parents=True, exist_ok=True)
    (input_root / "walk1.bvh").write_text("placeholder", encoding="utf-8")
    (input_root / "obstacle_run.bvh").write_text("placeholder", encoding="utf-8")
    nested = input_root / "subject1"
    nested.mkdir()
    (nested / "obstacle_jump.bvh").write_text("placeholder", encoding="utf-8")

    source = DatasetSourceSpec(
        name="lafan1",
        type="bvh",
        input=str(input_root),
        bvh_format="lafan1",
        exclude_patterns=("*obstacle*",),
    )

    items, _scan_root, report = dataset_builder._collect_source_files_with_report(
        source,
        quiet=True,
    )

    assert [item.rel_no_suffix.as_posix() for item in items] == ["walk1"]
    assert report["scanned_files"] == 3
    assert report["path_rejected_files"] == 2
    assert report["kept_files"] == 1
    assert report["filtered_files"] == 2
    assert report["path_reject_reasons"] == {"*obstacle*": 2}


def test_collect_source_files_with_report_preserves_path_excludes_with_metadata(tmp_path: Path) -> None:
    input_root = tmp_path / "seed_source" / "g1" / "csv"
    input_root.mkdir(parents=True, exist_ok=True)
    for name in ("walk_a.csv", "walk_b.csv", "obstacle_walk.csv"):
        (input_root / name).write_text("placeholder", encoding="utf-8")

    metadata_csv = tmp_path / "metadata.csv"
    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["move_g1_path", "is_mirror"])
        writer.writeheader()
        for name in ("walk_a.csv", "walk_b.csv", "obstacle_walk.csv"):
            writer.writerow({"move_g1_path": f"g1/csv/{name}", "is_mirror": "False"})

    source = DatasetSourceSpec(
        name="seed",
        type="seed_csv",
        input=str(input_root),
        metadata_csv=str(metadata_csv),
        filters={"is_mirror": [False]},
        exclude_patterns=("*obstacle*",),
    )

    items, _scan_root, report = dataset_builder._collect_source_files_with_report(
        source,
        quiet=True,
    )

    assert [item.rel_no_suffix.as_posix() for item in items] == ["walk_a", "walk_b"]
    assert report["scanned_files"] == 3
    assert report["path_rejected_files"] == 1
    assert report["kept_files"] == 2
    assert report["filtered_files"] == 1
    assert report["path_reject_reasons"] == {"*obstacle*": 1}


def test_build_dataset_from_spec_writes_shard_directories(tmp_path: Path) -> None:
    npz_input = tmp_path / "npz_source"
    _write_npz_from_pkl(npz_input / "clip_a.npz")
    _write_npz_from_pkl(npz_input / "clip_b.npz")

    spec = DatasetSpec(
        name="demo_dataset",
        target_fps=30,
        sources=[DatasetSourceSpec(name="npz_src", type="npz", input=str(npz_input))],
    )

    output_root = tmp_path / "datasets"
    stale_cache = output_root / "demo_dataset" / "clips" / "npz_src" / "clip_a.npz"
    _write_npz_from_pkl(stale_cache)
    stale_payload = dict(np.load(stale_cache, allow_pickle=True))
    stale_payload["root_pos"] = np.asarray(stale_payload["root_pos"], dtype=np.float32) + 100.0
    np.savez(stale_cache, **stale_payload)

    report = build_dataset_from_spec(spec, jobs=2, skip_fk_check=True, output_root=output_root)

    dataset_dir = output_root / "demo_dataset"
    assert report["dataset_dir"] == str(dataset_dir)
    assert not (dataset_dir / "clips").exists()
    assert (dataset_dir / "shard_000.h5").is_file()
    assert report["input_clips"] == 2

    with h5py.File(dataset_dir / "shard_000.h5", "r") as shard:
        assert "root_pos" in shard
        assert "root_quat_w" in shard
        assert "joint_pos" in shard
        assert "body_pos_w" not in shard
        assert float(shard["root_pos"][0, 2]) < 10.0


def test_build_dataset_from_spec_rejects_clips_root_source_without_deleting_input(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "datasets"
    source_dir = output_root / "demo_dataset" / "clips" / "npz_src"
    clip_path = source_dir / "clip_a.npz"
    _write_npz_from_pkl(clip_path)

    spec = DatasetSpec(
        name="demo_dataset",
        target_fps=30,
        sources=[DatasetSourceSpec(name="npz_src", type="npz", input=str(source_dir))],
    )

    with pytest.raises(ValueError, match="temporary clips directory"):
        build_dataset_from_spec(spec, jobs=1, skip_fk_check=True, output_root=output_root)

    assert clip_path.is_file()


def test_collect_clip_rows_ignores_stale_excluded_cached_npz(tmp_path: Path) -> None:
    npz_input = tmp_path / "npz_source"
    for name in ("keep_a.npz", "keep_b.npz", "obstacle_old.npz"):
        _write_npz_from_pkl(npz_input / name)

    spec = DatasetSpec(
        name="demo_dataset",
        target_fps=30,
        sources=[
            DatasetSourceSpec(
                name="npz_src",
                type="npz",
                input=str(npz_input),
                exclude_patterns=("*obstacle*",),
            )
        ],
    )
    paths = dataset_builder.resolve_dataset_paths(spec, output_root=tmp_path / "datasets")
    source_dir = paths.clips_root / "npz_src"
    for name in ("keep_a.npz", "keep_b.npz", "obstacle_old.npz"):
        _write_npz_from_pkl(source_dir / name)

    rows = dataset_builder.collect_clip_rows(spec, paths=paths)

    assert sorted(row.clip_id for row in rows) == ["npz_src:keep_a", "npz_src:keep_b"]


def test_convert_source_to_npz_clips_applies_preprocess(tmp_path: Path) -> None:
    npz_input = tmp_path / "npz_source"
    _write_npz_from_pkl(npz_input / "clip_a.npz")

    source = DatasetSourceSpec(name="npz_src", type="npz", input=str(npz_input))
    output_dir = tmp_path / "dataset" / "clips" / "npz_src"
    report = convert_source_to_npz_clips(
        source,
        output_dir,
        jobs=1,
        preprocess=DatasetSpec(
            name="unused",
            target_fps=30,
            sources=[source],
        ).preprocess,
    )

    assert report["clips"] == 1

    from train_mimic.data.preprocess import DatasetPreprocessSpec

    output_dir_2 = tmp_path / "dataset2" / "clips" / "npz_src"
    convert_source_to_npz_clips(
        source,
        output_dir_2,
        jobs=1,
        preprocess=DatasetPreprocessSpec(
            normalize_root_xy=True,
            ground_align="first_frame_foot",
        ),
    )
    clip = np.load(output_dir_2 / "clip_a.npz", allow_pickle=True)
    body_names = [str(name) for name in clip["body_names"].tolist()]
    pelvis_idx = body_names.index("pelvis")
    left_idx = body_names.index("left_ankle_roll_link")
    right_idx = body_names.index("right_ankle_roll_link")
    assert np.allclose(clip["body_pos_w"][0, pelvis_idx, :2], 0.0)
    foot_z = clip["body_pos_w"][:, [left_idx, right_idx], 2]
    assert np.isclose(float(np.min(foot_z[0])), 0.0)


def test_convert_source_to_npz_clips_skips_all_off_ground_clips_before_ground_align(tmp_path: Path) -> None:
    npz_input = tmp_path / "npz_source"
    _write_npz_from_pkl(npz_input / "keep.npz")
    _write_npz_from_pkl(npz_input / "float.npz")

    for name, floating in (("keep.npz", False), ("float.npz", True)):
        path = npz_input / name
        clip = dict(np.load(path, allow_pickle=True))
        body_names = [str(body_name) for body_name in clip["body_names"].tolist()]
        left_idx = body_names.index("left_ankle_roll_link")
        right_idx = body_names.index("right_ankle_roll_link")
        body_pos_w = np.asarray(clip["body_pos_w"]).copy()
        if floating:
            body_pos_w[..., 2] = np.maximum(body_pos_w[..., 2], 0.3)
        else:
            body_pos_w[:, left_idx, 2] = 0.0
            body_pos_w[:, right_idx, 2] = 0.3
        clip["body_pos_w"] = body_pos_w
        np.savez(path, **clip)

    source = DatasetSourceSpec(name="npz_src", type="npz", input=str(npz_input))
    output_dir = tmp_path / "dataset" / "clips" / "npz_src"
    report = convert_source_to_npz_clips(
        source,
        output_dir,
        jobs=1,
        preprocess=dataset_builder.DatasetPreprocessSpec(
            ground_align="first_frame_foot",
            max_all_off_ground_s=0.05,
            off_ground_height=0.08,
        ),
    )

    assert report["clips"] == 1
    assert (output_dir / "keep.npz").is_file()
    assert not (output_dir / "float.npz").exists()
    assert (output_dir / "float.npz.filtered.json").is_file()

    def _unexpected_run_conversion_tasks(*_args, **_kwargs):
        raise AssertionError("filtered clip should be skipped by marker on incremental rebuild")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(dataset_builder, "run_conversion_tasks", _unexpected_run_conversion_tasks)
    try:
        second_report = convert_source_to_npz_clips(
            source,
            output_dir,
            jobs=1,
            preprocess=dataset_builder.DatasetPreprocessSpec(
                ground_align="first_frame_foot",
                max_all_off_ground_s=0.05,
                off_ground_height=0.08,
            ),
        )
    finally:
        monkeypatch.undo()

    assert second_report["clips"] == 1


def test_merge_clip_dicts_rejects_inconsistent_body_names(tmp_path: Path) -> None:
    clip_a = tmp_path / "clip_a.npz"
    clip_b = tmp_path / "clip_b.npz"
    _write_npz_from_pkl(clip_a)
    _write_npz_from_pkl(clip_b)

    clip_a_dict = dict(np.load(clip_a, allow_pickle=True))
    clip_b_dict = dict(np.load(clip_b, allow_pickle=True))
    clip_b_dict["body_names"] = np.array(["wrong"] * len(clip_b_dict["body_names"]), dtype=str)

    with pytest.raises(ValueError, match="inconsistent body_names"):
        merge_clip_dicts([clip_a_dict, clip_b_dict], tmp_path / "merged.npz")


def test_merge_clip_dicts_rejects_non_positive_target_fps(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.npz"
    _write_npz_from_pkl(clip_path)
    clip_dict = dict(np.load(clip_path, allow_pickle=True))

    with pytest.raises(ValueError, match="target_fps must be > 0"):
        merge_clip_dicts([clip_dict], tmp_path / "merged.npz", target_fps=0)


def test_merge_clip_dicts_accepts_single_frame_clip(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip_single.npz"
    _write_npz_from_pkl(clip_path)
    clip_dict = dict(np.load(clip_path, allow_pickle=True))
    for key in [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]:
        clip_dict[key] = np.asarray(clip_dict[key])[:1]

    merge_clip_dicts([clip_dict], tmp_path / "merged_single.npz")

    merged = np.load(tmp_path / "merged_single.npz", allow_pickle=True)
    assert merged["clip_lengths"].tolist() == [1]


def test_batch_convert_chunk_preprocess_sees_resampled_target_fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkl_path = tmp_path / "clip.pkl"
    pkl_path.write_bytes(b"placeholder")

    clip_npz = tmp_path / "clip.npz"
    _write_npz_from_pkl(clip_npz)
    arrays = dict(np.load(clip_npz, allow_pickle=True))
    arrays["fps"] = 60

    observed_fps: list[int] = []

    monkeypatch.setattr(
        "train_mimic.scripts.convert_pkl_to_npz.convert_pkl_to_arrays",
        lambda *_args, **_kwargs: arrays.copy(),
    )

    def _capture(clip_dict, *, preprocess, clip_label):
        observed_fps.append(int(clip_dict["fps"]))
        return clip_dict

    monkeypatch.setattr(dataset_builder, "_maybe_preprocess_clip_dict", _capture)

    dataset_builder._batch_convert_chunk(
        [str(pkl_path)],
        30,
        str(tmp_path / "merged.npz"),
        "train",
        preprocess=dataset_builder.DatasetPreprocessSpec(),
    )

    assert observed_fps == [30]


def test_batch_convert_chunk_skips_filtered_short_clips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    short_path = tmp_path / "short.pkl"
    valid_path = tmp_path / "valid.pkl"
    short_path.write_bytes(b"placeholder")
    valid_path.write_bytes(b"placeholder")

    num_bodies = len(_MJLAB_G1_BODY_NAMES)

    def _arrays(num_frames: int) -> dict[str, object]:
        body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
        body_quat_w[..., 0] = 1.0
        return {
            "fps": 30,
            "root_pos": np.zeros((num_frames, 3), dtype=np.float32),
            "root_quat_w": np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_frames, 1)),
            "joint_pos": np.zeros((num_frames, 29), dtype=np.float32),
            "joint_vel": np.zeros((num_frames, 29), dtype=np.float32),
            "body_pos_w": np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
            "body_ang_vel_w": np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
            "body_names": np.asarray(_MJLAB_G1_BODY_NAMES, dtype=str),
        }

    def _convert(path: str, **_kwargs):
        if path.endswith("short.pkl"):
            return _arrays(18)
        if path.endswith("valid.pkl"):
            return _arrays(22)
        raise AssertionError(path)

    monkeypatch.setattr(
        "train_mimic.scripts.convert_pkl_to_npz.convert_pkl_to_arrays",
        _convert,
    )

    stats = dataset_builder._batch_convert_chunk(
        [str(short_path), str(valid_path)],
        30,
        str(tmp_path / "merged.h5"),
        "train",
        preprocess=dataset_builder.DatasetPreprocessSpec(
            normalize_root_xy=True,
            ground_align="first_frame_foot",
            min_frames=22,
        ),
    )

    assert stats["clips"] == 1
    assert stats["kept_file_paths"] == [str(valid_path)]
    with h5py.File(tmp_path / "merged.h5", "r") as merged:
        assert merged["clip_lengths"][()].tolist() == [22]


def test_shard_stats_counts_real_frames_not_overlapped_windows(tmp_path: Path) -> None:
    stats = dataset_builder._shard_stats(
        output_dir=tmp_path,
        shard_infos=[{
            "path": tmp_path / "shard_000.h5",
            "clips": 3,
            "frames": 1000,
            "clip_lengths": [512, 512, 512],
            "source_clip_lengths": [1000],
        }],
        fps=30,
    )

    assert stats["frames"] == 1000
    assert stats["duration_s"] == 1000 / 30


def test_compute_dataset_stats_reports_total_duration_from_source_clips(tmp_path: Path) -> None:
    num_bodies = len(_MJLAB_G1_BODY_NAMES)
    total_frames = 18
    body_quat_w = np.zeros((total_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    write_hdf5_motion_shard(
        {
            "fps": 30,
            "root_pos": np.zeros((total_frames, 3), dtype=np.float32),
            "root_quat_w": np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (total_frames, 1),
            ),
            "joint_pos": np.zeros((total_frames, 29), dtype=np.float32),
            "joint_vel": np.zeros((total_frames, 29), dtype=np.float32),
            "body_pos_w": np.zeros((total_frames, num_bodies, 3), dtype=np.float32),
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": np.zeros((total_frames, num_bodies, 3), dtype=np.float32),
            "body_ang_vel_w": np.zeros((total_frames, num_bodies, 3), dtype=np.float32),
            "body_names": np.asarray(_MJLAB_G1_BODY_NAMES, dtype=str),
            "clip_starts": np.asarray([0, 12], dtype=np.int64),
            "clip_lengths": np.asarray([12, 6], dtype=np.int64),
            "clip_fps": np.asarray([30, 60], dtype=np.int64),
        },
        tmp_path / "shard_000.h5",
        max_window_frames=8,
        overlap_frames=2,
    )

    stats = compute_dataset_stats(tmp_path)

    assert stats["frames"] == total_frames
    assert stats["duration_s"] == pytest.approx(12 / 30 + 6 / 60)
    assert stats["duration_h"] == pytest.approx((12 / 30 + 6 / 60) / 3600)
    assert stats["shard_details"][0]["duration_s"] == pytest.approx(12 / 30 + 6 / 60)



def test_build_dataset_batch_manifest_skips_filtered_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "seed_source"
    keep_train = source_dir / "keep_train.csv"
    drop_train = source_dir / "drop_train.csv"
    keep_val = source_dir / "keep_val.csv"
    for path in (keep_train, drop_train, keep_val):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")

    spec = DatasetSpec(
        name="seed_demo",
        target_fps=30,
        sources=[DatasetSourceSpec(name="seed", type="seed_csv", input=str(source_dir))],
    )
    dataset_dir = tmp_path / "datasets" / spec.name

    def _collect_with_report(_source, *, quiet=False):
        _ = quiet
        return ([
            SourceInputFile(path=keep_train, rel_no_suffix=Path("keep_train")),
            SourceInputFile(path=drop_train, rel_no_suffix=Path("drop_train")),
            SourceInputFile(path=keep_val, rel_no_suffix=Path("keep_val")),
        ], source_dir, {
            "source": "seed",
            "type": "seed_csv",
            "metadata_csv": None,
            "seed_filter_preset": "groot_strict",
            "scanned_files": 3,
            "metadata_rows_matched": 3,
            "preset_rejected_rows": 1,
            "kept_files": 2,
            "filtered_files": 1,
            "preset_reject_reasons": {"content_body_position:sitting": 1},
        })

    num_bodies = len(_MJLAB_G1_BODY_NAMES)

    def _write_merged(path: Path, lengths: list[int]) -> dict:
        total = sum(lengths)
        joint_pos = np.zeros((total, 29), dtype=np.float32)
        joint_vel = np.zeros_like(joint_pos)
        root_pos = np.zeros((total, 3), dtype=np.float32)
        root_quat_w = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (total, 1))
        body_pos_w = np.zeros((total, num_bodies, 3), dtype=np.float32)
        body_quat_w = np.zeros((total, num_bodies, 4), dtype=np.float32)
        body_quat_w[..., 0] = 1.0
        body_lin_vel_w = np.zeros((total, num_bodies, 3), dtype=np.float32)
        body_ang_vel_w = np.zeros((total, num_bodies, 3), dtype=np.float32)
        clip_lengths = np.asarray(lengths, dtype=np.int64)
        clip_starts = np.zeros(len(lengths), dtype=np.int64)
        if len(lengths) > 1:
            clip_starts[1:] = np.cumsum(clip_lengths[:-1])
        return write_hdf5_motion_shard({
            "fps": 30,
            "root_pos": root_pos,
            "root_quat_w": root_quat_w,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
            "body_names": np.asarray(_MJLAB_G1_BODY_NAMES, dtype=str),
            "clip_starts": clip_starts,
            "clip_lengths": clip_lengths,
            "clip_fps": np.full(len(lengths), 30, dtype=np.int64),
        }, path)

    def _batch_convert_split(clips, target_fps, output_dir, jobs, label, preprocess):
        _ = clips, target_fps, jobs, preprocess
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_path = output_dir / "shard_000.h5"
        h5_info = _write_merged(shard_path, [22, 24])
        return ({
            "output": str(output_dir),
            "shards": 1,
            "clips": 2,
            "num_clips": 2,
            "frames": 46,
            "fps": 30,
            "duration_s": 46.0 / 30.0,
        }, [{
            "path": shard_path,
            "clip_lengths": h5_info["clip_lengths"],
            "source_clip_lengths": h5_info["source_clip_lengths"],
            "frames": h5_info["frames"],
            "kept_file_paths": [str(keep_train), str(keep_val)],
        }])

    monkeypatch.setattr(dataset_builder, "_collect_source_files_with_report", _collect_with_report)
    monkeypatch.setattr(dataset_builder, "_batch_convert_split", _batch_convert_split)

    report = dataset_builder._build_dataset_batch(
        spec,
        paths=dataset_builder.resolve_dataset_paths(spec, output_root=tmp_path / "datasets"),
        force=False,
        skip_fk_check=True,
        skip_validate=False,
        jobs=2,
    )

    assert (dataset_dir / "shard_000.h5").is_file()
    assert not (dataset_dir / "manifest_resolved.csv").exists()
    assert report["input_clips"] == 3
    assert report["stats"]["clips"] == 2
    assert report["source_filters"][0]["seed_filter_preset"] == "groot_strict"
    assert report["source_filters"][0]["preset_reject_reasons"] == {"content_body_position:sitting": 1}


def test_build_dataset_batch_clears_stale_top_level_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "seed_source"
    keep_path = source_dir / "keep.csv"
    keep_path.parent.mkdir(parents=True, exist_ok=True)
    keep_path.write_text("placeholder", encoding="utf-8")

    spec = DatasetSpec(
        name="seed_demo",
        target_fps=30,
        sources=[DatasetSourceSpec(name="seed", type="seed_csv", input=str(source_dir))],
    )
    paths = dataset_builder.resolve_dataset_paths(spec, output_root=tmp_path / "datasets")
    paths.dataset_dir.mkdir(parents=True)
    stale_shard = paths.dataset_dir / "shard_999.h5"
    stale_tmp = paths.dataset_dir / ".seed_demo_chunk_7.h5"
    stale_shard.write_text("stale", encoding="utf-8")
    stale_tmp.write_text("stale", encoding="utf-8")

    def _collect_with_report(_source, *, quiet=False):
        _ = quiet
        return ([
            SourceInputFile(path=keep_path, rel_no_suffix=Path("keep")),
        ], source_dir, {
            "source": "seed",
            "type": "seed_csv",
            "metadata_csv": None,
            "seed_filter_preset": None,
            "scanned_files": 1,
            "metadata_rows_matched": 1,
            "kept_files": 1,
            "filtered_files": 0,
        })

    def _batch_convert_split(clips, target_fps, output_dir, jobs, label, preprocess):
        _ = clips, target_fps, jobs, label, preprocess
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_path = output_dir / "shard_000.h5"
        num_bodies = len(_MJLAB_G1_BODY_NAMES)
        h5_info = write_hdf5_motion_shard({
            "fps": 30,
            "root_pos": np.zeros((22, 3), dtype=np.float32),
            "root_quat_w": np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (22, 1)),
            "joint_pos": np.zeros((22, 29), dtype=np.float32),
            "joint_vel": np.zeros((22, 29), dtype=np.float32),
            "body_pos_w": np.zeros((22, num_bodies, 3), dtype=np.float32),
            "body_quat_w": np.tile(
                np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
                (22, num_bodies, 1),
            ),
            "body_lin_vel_w": np.zeros((22, num_bodies, 3), dtype=np.float32),
            "body_ang_vel_w": np.zeros((22, num_bodies, 3), dtype=np.float32),
            "body_names": np.asarray(_MJLAB_G1_BODY_NAMES, dtype=str),
            "clip_starts": np.asarray([0], dtype=np.int64),
            "clip_lengths": np.asarray([22], dtype=np.int64),
            "clip_fps": np.asarray([30], dtype=np.int64),
        }, shard_path)
        return ({
            "output": str(output_dir),
            "shards": 1,
            "clips": 1,
            "num_clips": 1,
            "frames": 22,
            "fps": 30,
            "duration_s": 22.0 / 30.0,
        }, [{
            "path": shard_path,
            "clip_lengths": h5_info["clip_lengths"],
            "source_clip_lengths": h5_info["source_clip_lengths"],
            "frames": h5_info["frames"],
            "kept_file_paths": [str(keep_path)],
        }])

    monkeypatch.setattr(dataset_builder, "_collect_source_files_with_report", _collect_with_report)
    monkeypatch.setattr(dataset_builder, "_batch_convert_split", _batch_convert_split)

    dataset_builder._build_dataset_batch(
        spec,
        paths=paths,
        force=False,
        skip_fk_check=True,
        skip_validate=False,
        jobs=1,
    )

    assert (paths.dataset_dir / "shard_000.h5").is_file()
    assert not stale_shard.exists()
    assert not stale_tmp.exists()
