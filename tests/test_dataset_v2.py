from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from train_mimic.data import dataset_builder
from train_mimic.data.dataset_builder import (
    DatasetClipRow,
    SourceInputFile,
    DatasetSourceSpec,
    DatasetSpec,
    assign_splits,
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
val_percent: 5
hash_salt: ""
sources:
  - name: clips
    type: npz
    input: {tmp_path / 'npz_source'}
    weight: 2.5
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
    assert spec.sources[0].weight == 2.5
    assert spec.sources[1].type == "bvh"
    assert spec.sources[1].bvh_format == "lafan1"


def test_load_dataset_spec_parses_preprocess(tmp_path: Path) -> None:
    spec_path = tmp_path / "preprocessed.yaml"
    spec_path.write_text(
        f"""name: demo
target_fps: 30
val_percent: 5
hash_salt: ""
preprocess:
  normalize_root_xy: true
  ground_align: clip_min_foot
  min_frames: 10
sources:
  - name: clips
    type: npz
    input: {tmp_path / 'npz_source'}
""",
        encoding="utf-8",
    )

    spec = load_dataset_spec(spec_path)
    assert spec.preprocess.normalize_root_xy is True
    assert spec.preprocess.ground_align == "clip_min_foot"
    assert spec.preprocess.min_frames == 10


def test_load_dataset_spec_rejects_bvh_without_format(tmp_path: Path) -> None:
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text(
        """name: demo
target_fps: 30
val_percent: 5
hash_salt: ""
sources:
  - name: broken
    type: bvh
    input: data/lafan1_bvh
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires bvh_format"):
        load_dataset_spec(spec_path)


def test_assign_splits_guarantees_non_empty_train_and_val() -> None:
    rows = [
        DatasetClipRow("clip_a", "src", "a.npz", 10, 30, "", "/tmp/a.npz"),
        DatasetClipRow("clip_b", "src", "b.npz", 10, 30, "", "/tmp/b.npz"),
    ]
    resolved = assign_splits(rows, 1, "")
    splits = {row.clip_id: row.resolved_split for row in resolved}
    assert set(splits.values()) == {"train", "val"}


def test_assign_splits_rejects_single_clip_dataset() -> None:
    rows = [DatasetClipRow("clip_a", "src", "a.npz", 10, 30, "", "/tmp/a.npz")]
    with pytest.raises(ValueError, match="at least 2 clips"):
        assign_splits(rows, 5, "")


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


def test_build_dataset_from_spec_writes_shard_directories(tmp_path: Path) -> None:
    npz_input = tmp_path / "npz_source"
    _write_npz_from_pkl(npz_input / "clip_a.npz")
    _write_npz_from_pkl(npz_input / "clip_b.npz")

    spec = DatasetSpec(
        name="demo_dataset",
        target_fps=30,
        val_percent=5,
        hash_salt="",
        sources=[DatasetSourceSpec(name="npz_src", type="npz", input=str(npz_input))],
    )

    output_root = tmp_path / "datasets"
    report = build_dataset_from_spec(spec, jobs=2, skip_fk_check=True, output_root=output_root)

    dataset_dir = output_root / "demo_dataset"
    assert report["dataset_dir"] == str(dataset_dir)
    assert report["build_dir"] == str(dataset_dir)
    assert (dataset_dir / "clips" / "npz_src" / "clip_a.npz").is_file()
    assert (dataset_dir / "clips" / "npz_src" / "clip_b.npz").is_file()
    assert (dataset_dir / "train" / "shard_000.npz").is_file()
    assert (dataset_dir / "val" / "shard_000.npz").is_file()
    assert (dataset_dir / "manifest_resolved.csv").is_file()
    assert (dataset_dir / "build_info.json").is_file()
    assert report["clip_counts"]["total"] == 2

    train_data = np.load(dataset_dir / "train" / "shard_000.npz", allow_pickle=True)
    assert "clip_starts" in train_data.files
    assert "clip_lengths" in train_data.files


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
            val_percent=5,
            hash_salt="",
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
            ground_align="clip_min_foot",
        ),
    )
    clip = np.load(output_dir_2 / "clip_a.npz", allow_pickle=True)
    body_names = [str(name) for name in clip["body_names"].tolist()]
    pelvis_idx = body_names.index("pelvis")
    left_idx = body_names.index("left_ankle_roll_link")
    right_idx = body_names.index("right_ankle_roll_link")
    assert np.allclose(clip["body_pos_w"][0, pelvis_idx, :2], 0.0)
    foot_z = clip["body_pos_w"][:, [left_idx, right_idx], 2]
    assert np.isclose(float(np.min(foot_z)), 0.0)


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
        [1.0],
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
        [1.0, 1.0],
        30,
        str(tmp_path / "merged.npz"),
        "train",
        preprocess=dataset_builder.DatasetPreprocessSpec(
            normalize_root_xy=True,
            ground_align="clip_min_foot",
            min_frames=22,
        ),
    )

    merged = np.load(tmp_path / "merged.npz", allow_pickle=True)
    assert stats["clips"] == 1
    assert stats["kept_file_paths"] == [str(valid_path)]
    assert merged["clip_lengths"].tolist() == [22]



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
        val_percent=5,
        hash_salt="",
        sources=[DatasetSourceSpec(name="seed", type="seed_csv", input=str(source_dir))],
    )
    dataset_dir = tmp_path / "datasets" / spec.name

    def _collect(_source):
        return ([
            SourceInputFile(path=keep_train, rel_no_suffix=Path("keep_train")),
            SourceInputFile(path=drop_train, rel_no_suffix=Path("drop_train")),
            SourceInputFile(path=keep_val, rel_no_suffix=Path("keep_val")),
        ], source_dir)

    def _hash_split(clip_id: str, _val_percent: int, _salt: str = "") -> str:
        return "val" if clip_id.endswith("keep_val") else "train"

    num_bodies = len(_MJLAB_G1_BODY_NAMES)

    def _write_merged(path: Path, lengths: list[int]) -> None:
        total = sum(lengths)
        joint_pos = np.zeros((total, 29), dtype=np.float32)
        joint_vel = np.zeros_like(joint_pos)
        body_pos_w = np.zeros((total, num_bodies, 3), dtype=np.float32)
        body_quat_w = np.zeros((total, num_bodies, 4), dtype=np.float32)
        body_quat_w[..., 0] = 1.0
        body_lin_vel_w = np.zeros((total, num_bodies, 3), dtype=np.float32)
        body_ang_vel_w = np.zeros((total, num_bodies, 3), dtype=np.float32)
        clip_lengths = np.asarray(lengths, dtype=np.int64)
        clip_starts = np.zeros(len(lengths), dtype=np.int64)
        if len(lengths) > 1:
            clip_starts[1:] = np.cumsum(clip_lengths[:-1])
        np.savez(
            path,
            fps=30,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            body_names=np.asarray(_MJLAB_G1_BODY_NAMES, dtype=str),
            clip_starts=clip_starts,
            clip_lengths=clip_lengths,
            clip_fps=np.full(len(lengths), 30, dtype=np.int64),
            clip_weights=np.ones(len(lengths), dtype=np.float64),
        )

    def _batch_convert_split(clips, target_fps, output_dir, jobs, split_name, preprocess):
        _ = clips, target_fps, jobs, preprocess
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_path = output_dir / "shard_000.npz"
        if split_name == "train":
            _write_merged(shard_path, [22])
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
                "clip_lengths": [22],
                "kept_file_paths": [str(keep_train)],
            }])
        _write_merged(shard_path, [24])
        return ({
            "output": str(output_dir),
            "shards": 1,
            "clips": 1,
            "num_clips": 1,
            "frames": 24,
            "fps": 30,
            "duration_s": 24.0 / 30.0,
        }, [{
            "path": shard_path,
            "clip_lengths": [24],
            "kept_file_paths": [str(keep_val)],
        }])

    monkeypatch.setattr(dataset_builder, "_collect_source_files", _collect)
    monkeypatch.setattr(dataset_builder, "hash_split", _hash_split)
    monkeypatch.setattr(dataset_builder, "_batch_convert_split", _batch_convert_split)

    report = dataset_builder._build_dataset_batch(
        spec,
        paths=dataset_builder.resolve_dataset_paths(spec, output_root=tmp_path / "datasets"),
        force=False,
        skip_fk_check=True,
        skip_validate=False,
        jobs=2,
    )

    manifest = (dataset_dir / "manifest_resolved.csv").read_text(encoding="utf-8")
    assert "seed:keep_train" in manifest
    assert "seed:keep_val" in manifest
    assert "seed:drop_train" not in manifest
    assert report["clip_counts"] == {"total": 2, "train": 1, "val": 1}
    assert report["input_clip_counts"] == {"total": 3, "train": 2, "val": 1}
