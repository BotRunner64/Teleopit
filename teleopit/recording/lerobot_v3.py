"""LeRobot v3 adapter and schema helpers for Teleopit sim2real recording."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from teleopit.constants import FULL_QPOS_DIM, NUM_JOINTS
from teleopit.controllers.observation import _quat_rotate_np
from teleopit.math_utils import quat_inv_np
from teleopit.runtime.common import cfg_get


IMAGE_KEY = "observation.images.d435i_rgb"
STATE_KEY = "observation.state"
MODE_KEY = "observation.mode"
ACTION_KEY = "action"
HAND_ACTION_KEY = "action.hand"
STATE_DIM = 68
MODE_DIM = 1
ACTION_DIM = FULL_QPOS_DIM
HAND_ACTION_DIM = 12
DEFAULT_IMAGE_SHAPE = (480, 640, 3)
MODE_CODES = {
    "standing": 0,
    "mocap": 1,
    "arms": 2,
    "pause": 3,
}


def _import_lerobot_dataset() -> Any:
    try:
        module = importlib.import_module("lerobot.datasets.lerobot_dataset")
    except ModuleNotFoundError as new_exc:
        if new_exc.name not in {"lerobot", "lerobot.datasets", "lerobot.datasets.lerobot_dataset"}:
            raise RuntimeError("Failed to import LeRobotDataset from the current LeRobot API") from new_exc
        try:
            module = importlib.import_module("lerobot.common.datasets.lerobot_dataset")
        except ModuleNotFoundError as old_exc:
            if old_exc.name not in {
                "lerobot",
                "lerobot.common",
                "lerobot.common.datasets",
                "lerobot.common.datasets.lerobot_dataset",
            }:
                raise RuntimeError("Failed to import LeRobotDataset from the legacy LeRobot API") from old_exc
            raise RuntimeError(
                "recording.enabled=true requires a LeRobot version that provides "
                "LeRobotDataset. Install Teleopit with the recording extra, for example: "
                "pip install -e '.[recording]'."
            ) from old_exc
        except Exception as old_exc:
            raise RuntimeError("Failed to import LeRobotDataset from the legacy LeRobot API") from old_exc
    except Exception as new_exc:
        raise RuntimeError("Failed to import LeRobotDataset from the current LeRobot API") from new_exc
    return module.LeRobotDataset


@dataclass(frozen=True)
class RecordingSchema:
    image_key: str
    image_shape: tuple[int, int, int]
    state_key: str = STATE_KEY
    state_dim: int = STATE_DIM
    mode_key: str = MODE_KEY
    mode_dim: int = MODE_DIM
    action_key: str = ACTION_KEY
    action_dim: int = ACTION_DIM
    hand_action_key: str = HAND_ACTION_KEY
    hand_action_dim: int = HAND_ACTION_DIM


def build_recording_schema(camera_cfg: Any) -> RecordingSchema:
    key = str(cfg_get(camera_cfg, "key", IMAGE_KEY))
    width = int(cfg_get(camera_cfg, "width", DEFAULT_IMAGE_SHAPE[1]))
    height = int(cfg_get(camera_cfg, "height", DEFAULT_IMAGE_SHAPE[0]))
    if width <= 0 or height <= 0:
        raise ValueError("recording.camera.width and recording.camera.height must be positive")
    return RecordingSchema(image_key=key, image_shape=(height, width, 3))


def lerobot_features(schema: RecordingSchema) -> dict[str, dict[str, object]]:
    return {
        schema.image_key: {
            "dtype": "video",
            "shape": schema.image_shape,
            "names": ["height", "width", "channel"],
        },
        schema.state_key: {
            "dtype": "float32",
            "shape": (schema.state_dim,),
            "names": ["state"],
        },
        schema.mode_key: {
            "dtype": "float32",
            "shape": (schema.mode_dim,),
            "names": ["mode"],
        },
        schema.action_key: {
            "dtype": "float32",
            "shape": (schema.action_dim,),
            "names": ["action"],
        },
        schema.hand_action_key: {
            "dtype": "float32",
            "shape": (schema.hand_action_dim,),
            "names": ["hand_action"],
        },
    }


def modality_sidecar(schema: RecordingSchema) -> dict[str, object]:
    return {
        "version": 1,
        "features": {
            schema.image_key: {
                "type": "video",
                "shape": list(schema.image_shape),
                "dtype": "uint8",
            },
            schema.state_key: {
                "type": "low_dim",
                "shape": [schema.state_dim],
                "dtype": "float32",
                "slices": {
                    "joint_pos": [0, 29],
                    "joint_vel": [29, 58],
                    "base_quat_wxyz": [58, 62],
                    "base_ang_vel": [62, 65],
                    "projected_gravity": [65, 68],
                },
            },
            schema.mode_key: {
                "type": "categorical",
                "shape": [schema.mode_dim],
                "dtype": "float32",
                "codes": MODE_CODES,
            },
            schema.action_key: {
                "type": "low_dim",
                "shape": [schema.action_dim],
                "dtype": "float32",
                "slices": {
                    "root_pos": [0, 3],
                    "root_quat_wxyz": [3, 7],
                    "joint_pos": [7, 36],
                },
            },
            schema.hand_action_key: {
                "type": "low_dim",
                "shape": [schema.hand_action_dim],
                "dtype": "float32",
                "units": "linkerhand_uint8_pose",
                "slices": {
                    "left_pose": [0, 6],
                    "right_pose": [6, 12],
                },
            },
        },
    }


def build_observation_state(robot_state: object) -> np.ndarray:
    joint_pos = np.asarray(getattr(robot_state, "qpos"), dtype=np.float32).reshape(-1)[:NUM_JOINTS]
    joint_vel = np.asarray(getattr(robot_state, "qvel"), dtype=np.float32).reshape(-1)[:NUM_JOINTS]
    base_quat = np.asarray(getattr(robot_state, "quat"), dtype=np.float32).reshape(-1)[:4]
    base_ang_vel = np.asarray(getattr(robot_state, "ang_vel"), dtype=np.float32).reshape(-1)[:3]
    if joint_pos.shape[0] != NUM_JOINTS:
        raise ValueError(f"robot_state.qpos must contain {NUM_JOINTS} joints, got {joint_pos.shape[0]}")
    if joint_vel.shape[0] != NUM_JOINTS:
        raise ValueError(f"robot_state.qvel must contain {NUM_JOINTS} joints, got {joint_vel.shape[0]}")
    if base_quat.shape[0] != 4:
        raise ValueError(f"robot_state.quat must be 4D (wxyz), got {base_quat.shape[0]}")
    if base_ang_vel.shape[0] != 3:
        raise ValueError(f"robot_state.ang_vel must be 3D, got {base_ang_vel.shape[0]}")
    gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    projected_gravity = _quat_rotate_np(quat_inv_np(base_quat), gravity_w)
    state = np.concatenate(
        [joint_pos, joint_vel, base_quat, base_ang_vel, projected_gravity],
        dtype=np.float32,
    )
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"recording observation.state must be {STATE_DIM}D, got {state.shape[0]}")
    return state


def normalize_action_reference_qpos(reference_qpos: object) -> np.ndarray:
    action = np.asarray(reference_qpos, dtype=np.float32).reshape(-1)[:ACTION_DIM]
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"recording action reference qpos must be {ACTION_DIM}D, got {action.shape[0]}")
    return action


def normalize_hand_action(left_pose: object, right_pose: object) -> np.ndarray:
    left = np.asarray(left_pose, dtype=np.float32).reshape(-1)
    right = np.asarray(right_pose, dtype=np.float32).reshape(-1)
    if left.shape[0] != 6:
        raise ValueError(f"recording left hand pose must be 6D, got {left.shape[0]}")
    if right.shape[0] != 6:
        raise ValueError(f"recording right hand pose must be 6D, got {right.shape[0]}")
    action = np.concatenate([left, right], dtype=np.float32)
    if action.shape[0] != HAND_ACTION_DIM:
        raise ValueError(f"recording action.hand must be {HAND_ACTION_DIM}D, got {action.shape[0]}")
    return action


def build_mode_observation(mode: str) -> np.ndarray:
    normalized = str(mode).strip().lower()
    if normalized not in MODE_CODES:
        raise ValueError(f"Unsupported recording mode {mode!r}; expected one of {sorted(MODE_CODES)}")
    return np.array([MODE_CODES[normalized]], dtype=np.float32)


class TeleopitLeRobotV3Recorder:
    """Small adapter around LeRobot v3 dataset writing."""

    def __init__(
        self,
        *,
        dataset: Any,
        output_dir: Path,
        schema: RecordingSchema,
    ) -> None:
        self._dataset = dataset
        self._output_dir = output_dir
        self._schema = schema
        self._active = False
        self._frames_in_episode = 0

    @classmethod
    def create(
        cls,
        *,
        output_dir: str | Path,
        dataset_name: str | None,
        repo_id: str | None,
        task: str,
        fps: int,
        schema: RecordingSchema,
    ) -> "TeleopitLeRobotV3Recorder":
        LeRobotDataset = _import_lerobot_dataset()

        root = Path(output_dir)
        root.parent.mkdir(parents=True, exist_ok=True)
        dataset_repo_id = repo_id or dataset_name or "teleopit/sim2real"
        features = lerobot_features(schema)

        try:
            dataset = LeRobotDataset.create(
                repo_id=dataset_repo_id,
                fps=int(fps),
                root=root,
                features=features,
                use_videos=True,
            )
        except TypeError:
            dataset = LeRobotDataset.create(
                repo_id=dataset_repo_id,
                fps=int(fps),
                root=root,
                features=features,
            )
        recorder = cls(dataset=dataset, output_dir=root, schema=schema)
        recorder._write_modality_sidecar()
        return recorder

    def start_episode(self) -> None:
        if self._active:
            raise RuntimeError("Cannot start a new recording episode while one is active")
        self._active = True
        self._frames_in_episode = 0

    def add_frame(
        self,
        *,
        image: np.ndarray,
        state: np.ndarray,
        mode: np.ndarray,
        action: np.ndarray,
        hand_action: np.ndarray,
        task: str,
    ) -> None:
        if not self._active:
            raise RuntimeError("Cannot add a recording frame without an active episode")
        image_arr = np.asarray(image, dtype=np.uint8)
        if tuple(image_arr.shape) != self._schema.image_shape:
            raise ValueError(f"{self._schema.image_key} frame shape {image_arr.shape} != {self._schema.image_shape}")
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        mode_arr = np.asarray(mode, dtype=np.float32).reshape(-1)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        hand_action_arr = np.asarray(hand_action, dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != self._schema.state_dim:
            raise ValueError(f"{self._schema.state_key} must be {self._schema.state_dim}D")
        if mode_arr.shape[0] != self._schema.mode_dim:
            raise ValueError(f"{self._schema.mode_key} must be {self._schema.mode_dim}D")
        if action_arr.shape[0] != self._schema.action_dim:
            raise ValueError(f"{self._schema.action_key} must be {self._schema.action_dim}D")
        if hand_action_arr.shape[0] != self._schema.hand_action_dim:
            raise ValueError(f"{self._schema.hand_action_key} must be {self._schema.hand_action_dim}D")
        self._dataset.add_frame(
            {
                self._schema.image_key: image_arr,
                self._schema.state_key: state_arr,
                self._schema.mode_key: mode_arr,
                self._schema.action_key: action_arr,
                self._schema.hand_action_key: hand_action_arr,
                "task": str(task),
            }
        )
        self._frames_in_episode += 1

    def save_episode(self) -> None:
        if not self._active:
            return
        self._dataset.save_episode()
        self._active = False
        self._frames_in_episode = 0

    def discard_episode(self) -> None:
        if not self._active:
            return
        clear = getattr(self._dataset, "clear_episode_buffer", None)
        if callable(clear):
            clear()
        else:
            buffer_attr = getattr(self._dataset, "episode_buffer", None)
            if isinstance(buffer_attr, dict):
                buffer_attr.clear()
        self._active = False
        self._frames_in_episode = 0

    def finalize(self) -> None:
        finalize = getattr(self._dataset, "finalize", None)
        if callable(finalize):
            finalize()
            return
        consolidate = getattr(self._dataset, "consolidate", None)
        if callable(consolidate):
            consolidate()

    def _write_modality_sidecar(self) -> None:
        path = self._output_dir / "meta" / "modality.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(modality_sidecar(self._schema), indent=2) + "\n", encoding="utf-8")
