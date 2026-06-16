"""Tests for train_mimic.scripts.train multi-GPU launcher helpers."""

from __future__ import annotations

import argparse
from functools import partial
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from train_mimic.app import DEFAULT_TASK, validate_checkpoint_path, validate_motion_file
from train_mimic.data.dataset_lib import PRECOMPUTED_MOTION_VERSION
from train_mimic.scripts import train
from train_mimic.tasks.tracking.config.rl import make_general_tracking_ppo_runner_cfg


class _CudaStub:
    @staticmethod
    def is_available() -> bool:
        return True


class _TorchStub:
    cuda = _CudaStub()


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "num_envs": 1024,
        "max_iterations": 10,
        "seed": 42,
        "logger": "tensorboard",
        "experiment_name": None,
        "motion_file": "data/datasets/seed_precomputed",
        "robot_xml": None,
        "resume": None,
        "sampling_mode": None,
        "rewind_prob": None,
        "rewind_min_steps": None,
        "rewind_max_steps": None,
        "cache_num_clips": None,
        "cache_swap_interval_steps": None,
        "device": None,
        "gpu_ids": None,
        "master_port": 29500,
        "video": False,
        "video_interval": 2000,
        "video_length": 200,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestTrainLauncherHelpers:
    def test_parse_args_defaults_to_tensorboard_logger(self) -> None:
        args = train.parse_args([])
        assert args.logger == "tensorboard"

    def test_parse_args_accepts_logger_choice(self) -> None:
        args = train.parse_args(["--logger", "swanlab"])
        assert args.logger == "swanlab"

    def test_parse_args_rejects_removed_wandb_project(self) -> None:
        with pytest.raises(SystemExit):
            train.parse_args(["--wandb_project", "teleopit"])

    def test_parse_args_with_gpu_ids(self) -> None:
        args = train.parse_args(["--gpu_ids", "0", "2", "3", "--master_port", "29600"])
        assert args.gpu_ids == [0, 2, 3]
        assert args.master_port == 29600

    def test_parse_args_accepts_robot_xml(self) -> None:
        args = train.parse_args(["--robot_xml", "assets/robots/unitree_g1/g1_29dof_dex3.xml"])
        assert args.robot_xml == "assets/robots/unitree_g1/g1_29dof_dex3.xml"

    def test_should_launch_multi_gpu(self) -> None:
        args = _args(gpu_ids=[0, 1, 2, 3])
        assert train._should_launch_multi_gpu(args, env={"WORLD_SIZE": "1"}) is True
        assert train._should_launch_multi_gpu(args, env={"WORLD_SIZE": "4"}) is False

    def test_validate_multi_gpu_args_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            train._validate_multi_gpu_args(_args(gpu_ids=[0, 1, 1]))

    def test_filtered_argv_for_worker_removes_launcher_only_flags(self) -> None:
        argv = [
            "train.py",
            "--gpu_ids",
            "0",
            "1",
            "2",
            "3",
            "--master_port",
            "29600",
            "--num_envs",
            "1024",
        ]
        assert train._filtered_argv_for_worker(argv) == [
            "train.py",
            "--num_envs",
            "1024",
        ]

    def test_build_torchrun_command_uses_visible_gpu_count(self) -> None:
        args = _args(gpu_ids=[0, 2, 5], master_port=29600)
        argv = ["train.py", "--gpu_ids", "0", "2", "5", "--num_envs", "1024"]
        command = train._build_torchrun_command(args, argv)
        assert command[:4] == [
            "torchrun",
            "--standalone",
            "--nproc_per_node=3",
            "--master_port=29600",
        ]
        assert command[4:] == ["train.py", "--num_envs", "1024"]

    def test_build_launcher_env_sets_cuda_visible_devices(self) -> None:
        env = train._build_launcher_env(_args(gpu_ids=[3, 1]), env={"PATH": "/bin"})
        assert env["CUDA_VISIBLE_DEVICES"] == "3,1"
        assert env["PATH"] == "/bin"

    def test_resolve_device_defaults_to_local_rank_in_distributed_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert train._resolve_device(_args(), _TorchStub()) == "cuda:2"

    def test_resolve_device_rejects_wrong_distributed_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_RANK", "1")
        with pytest.raises(ValueError, match="LOCAL_RANK=1"):
            train._resolve_device(_args(device="cuda:0"), _TorchStub())

    def test_resolve_worker_seed_offsets_by_global_rank(self) -> None:
        assert train._resolve_worker_seed(42, env={"WORLD_SIZE": "4", "RANK": "0"}) == 42
        assert train._resolve_worker_seed(42, env={"WORLD_SIZE": "4", "RANK": "3"}) == 300051

    def test_resolve_worker_seed_defaults_to_base_seed_without_rank(self) -> None:
        assert train._resolve_worker_seed(123, env={}) == 123

    def test_resolve_worker_seed_ignores_rank_outside_distributed_mode(self) -> None:
        assert train._resolve_worker_seed(42, env={"WORLD_SIZE": "1", "RANK": "3"}) == 42

    def test_configure_tensorboard_logger(self) -> None:
        agent_cfg = types.SimpleNamespace(logger="wandb", experiment_name="exp")
        env_cfg = types.SimpleNamespace()

        active = train._configure_experiment_logger(
            logger_name="tensorboard",
            agent_cfg=agent_cfg,
            env_cfg=env_cfg,
            log_dir="/tmp/run",
        )

        assert active is False
        assert agent_cfg.logger == "tensorboard"

    def test_configure_wandb_logger_uses_experiment_name_as_project(self) -> None:
        agent_cfg = types.SimpleNamespace(logger="tensorboard", experiment_name="exp")
        env_cfg = types.SimpleNamespace()

        active = train._configure_experiment_logger(
            logger_name="wandb",
            agent_cfg=agent_cfg,
            env_cfg=env_cfg,
            log_dir="/tmp/run",
        )

        assert active is False
        assert agent_cfg.logger == "wandb"
        assert agent_cfg.wandb_project == "exp"

    def test_configure_swanlab_logger_syncs_tensorboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, object]] = []
        fake_swanlab = types.SimpleNamespace(
            init=lambda **kwargs: calls.append(("init", kwargs)),
            sync_tensorboard_torch=lambda **kwargs: calls.append(("sync", kwargs)),
        )
        monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)
        monkeypatch.setenv("RANK", "0")
        agent_cfg = types.SimpleNamespace(logger="wandb", experiment_name="exp", max_iterations=10)
        env_cfg = types.SimpleNamespace(
            commands={
                "motion": types.SimpleNamespace(
                    motion_file="data/train",
                    sampling_mode="uniform",
                    rewind_prob=0.8,
                    rewind_min_steps=25,
                    rewind_max_steps=75,
                )
            },
            scene=types.SimpleNamespace(num_envs=64),
            robot_xml="/tmp/g1.xml",
        )

        active = train._configure_experiment_logger(
            logger_name="swanlab",
            agent_cfg=agent_cfg,
            env_cfg=env_cfg,
            log_dir="/tmp/2026-01-01_00-00-00",
        )

        assert active is True
        assert agent_cfg.logger == "tensorboard"
        assert calls == [
            (
                "init",
                {
                    "project": "exp",
                    "name": "2026-01-01_00-00-00",
                    "log_dir": "/tmp/2026-01-01_00-00-00",
                    "config": {
                        "experiment_name": "exp",
                        "motion_file": "data/train",
                        "robot_xml": "/tmp/g1.xml",
                        "num_envs": 64,
                        "max_iterations": 10,
                        "sampling_mode": "uniform",
                        "rewind_prob": 0.8,
                        "rewind_min_steps": 25,
                        "rewind_max_steps": 75,
                    },
                },
            ),
            ("sync", {"types": ["scalar", "scalars", "image", "text"]}),
        ]

    def test_main_uses_launcher_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: dict[str, object] = {}

        def fake_launch(args: argparse.Namespace, argv: list[str]) -> None:
            called["gpu_ids"] = args.gpu_ids
            called["argv"] = argv

        def fake_worker(args: argparse.Namespace) -> None:
            raise AssertionError("worker should not run in launcher branch")

        monkeypatch.setattr(train, "_launch_multi_gpu", fake_launch)
        monkeypatch.setattr(train, "_run_worker", fake_worker)

        train.main(["train.py", "--gpu_ids", "0", "1", "--num_envs", "1024"])

        assert called == {
            "gpu_ids": [0, 1],
            "argv": ["train.py", "--gpu_ids", "0", "1", "--num_envs", "1024"],
        }


def test_tracking_runner_configs_disable_model_upload() -> None:
    assert make_general_tracking_ppo_runner_cfg().upload_model is False


def test_make_g1_training_robot_cfg_uses_requested_xml() -> None:
    from train_mimic.tasks.tracking.config.env import make_g1_training_robot_cfg

    xml_path = Path("assets/robots/unitree_g1/g1_29dof_dex3.xml").resolve()
    robot_cfg = make_g1_training_robot_cfg(xml_path)

    assert isinstance(robot_cfg.spec_fn, partial)
    assert robot_cfg.spec_fn.args == (xml_path,)


def test_validate_motion_file_accepts_shard_directories(tmp_path: Path) -> None:
    num_frames = 3
    shard_path = tmp_path / "shard_000.h5"
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(shard_path, "w") as h5:
        h5.attrs["format"] = "teleopit_precomputed_motion_hdf5"
        h5.attrs["version"] = PRECOMPUTED_MOTION_VERSION
        h5.create_dataset("body_names", data=np.asarray(["pelvis"], dtype=object), dtype=str_dt)
        h5.create_dataset("joint_pos", data=np.zeros((num_frames, 29), dtype=np.float32), chunks=True)
        h5.create_dataset("joint_vel", data=np.zeros((num_frames, 29), dtype=np.float32), chunks=True)
        h5.create_dataset("body_pos_w", data=np.zeros((num_frames, 1, 3), dtype=np.float32), chunks=True)
        h5.create_dataset(
            "body_quat_w",
            data=np.tile(
                np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
                (num_frames, 1, 1),
            ),
            chunks=True,
        )
        h5.create_dataset("body_lin_vel_w", data=np.zeros((num_frames, 1, 3), dtype=np.float32), chunks=True)
        h5.create_dataset("body_ang_vel_w", data=np.zeros((num_frames, 1, 3), dtype=np.float32), chunks=True)
        h5.create_dataset("clip_starts", data=np.asarray([0], dtype=np.int64))
        h5.create_dataset("clip_lengths", data=np.asarray([num_frames], dtype=np.int64))
        h5.create_dataset("clip_fps", data=np.asarray([30], dtype=np.int64))
        h5.create_dataset("source_clip_ids", data=np.asarray([0], dtype=np.int64))
        h5.create_dataset("source_start_frames", data=np.asarray([0], dtype=np.int64))
        h5.create_dataset("source_clip_starts", data=np.asarray([0], dtype=np.int64))
        h5.create_dataset("source_clip_lengths", data=np.asarray([num_frames], dtype=np.int64))
        h5.create_dataset("source_clip_fps", data=np.asarray([30], dtype=np.int64))
    validate_motion_file(str(tmp_path))


def test_validate_motion_file_rejects_non_shard_paths(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Motion dataset not found"):
        validate_motion_file(str(tmp_path))


def test_validate_checkpoint_path_rejects_directories(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        validate_checkpoint_path(str(tmp_path))
