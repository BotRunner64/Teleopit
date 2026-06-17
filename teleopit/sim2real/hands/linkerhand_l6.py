from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import importlib.util
import logging
from typing import Any, Sequence

import numpy as np

from teleopit.runtime.common import cfg_get
from teleopit.sim2real.hands.base import HAND_SIDES, HandDevice, HandInputMapper, HandPoseCommand
from teleopit.sim2real.hands.pico_landmarks import pico_hand_to_landmarks

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOMEHAND_CONFIG = "third_party/somehand/configs/retargeting/bihand/linkerhand_l6_bihand.yaml"
DEFAULT_LINKERHAND_SDK_ROOT = "third_party/linkerhand-python-sdk"
THUMB_YAW_DEFAULT = 10
OPEN_POSE = (250, THUMB_YAW_DEFAULT, 250, 250, 250, 250)
CLOSE_POSE = (79, THUMB_YAW_DEFAULT, 0, 0, 0, 0)
DEFAULT_SPEED = (50, 50, 50, 50, 50, 50)
VR_HAND_POSE_SPEED = (255, 255, 255, 255, 255, 255)
L6_SDK_JOINT_ORDER = (
    "thumb_cmc_pitch",
    "thumb_cmc_roll",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)


@dataclass(frozen=True)
class LinkerHandL6Config:
    mode: str
    sides: tuple[str, ...]
    left_can: str
    right_can: str
    modbus: str
    rate_hz: float
    frame_timeout_s: float
    trigger_deadzone: float
    deadman_threshold: float
    thumb_yaw_center: int
    speed: tuple[int, ...]
    open_pose: tuple[int, ...]
    close_pose: tuple[int, ...]
    fixed_thumb_yaw: int | None
    print_input: bool
    somehand_config_path: str
    somehand_rate_hz: float
    somehand_max_iterations: int | None
    somehand_temporal_filter_alpha: float | None
    somehand_output_alpha: float | None


def parse_linkerhand_l6_config(cfg: Any) -> LinkerHandL6Config:
    hands_cfg = cfg_get(cfg, "hands", {}) or {}
    l6_cfg = cfg_get(hands_cfg, "linkerhand_l6", {}) or {}
    somehand_cfg = cfg_get(hands_cfg, "somehand", {}) or {}
    mode = str(cfg_get(hands_cfg, "mode", "gripper")).strip().lower()
    if mode not in ("gripper", "vr_hand_pose"):
        raise ValueError(f"hands.mode must be gripper or vr_hand_pose, got {mode!r}")
    sides = tuple(str(side).strip().lower() for side in cfg_get(hands_cfg, "sides", HAND_SIDES))
    if not sides or any(side not in HAND_SIDES for side in sides):
        raise ValueError("hands.sides must contain left, right, or both sides")
    thumb_yaw = _uint8(cfg_get(l6_cfg, "thumb_yaw_center", THUMB_YAW_DEFAULT), "thumb_yaw_center")
    open_pose = _pose_values(cfg_get(l6_cfg, "open_pose", OPEN_POSE), "open_pose")
    close_pose = _pose_values(cfg_get(l6_cfg, "close_pose", CLOSE_POSE), "close_pose")
    open_pose[1] = thumb_yaw
    close_pose[1] = thumb_yaw
    speed = VR_HAND_POSE_SPEED if mode == "vr_hand_pose" else tuple(_pose_values(cfg_get(l6_cfg, "speed", DEFAULT_SPEED), "speed"))
    return LinkerHandL6Config(
        mode=mode,
        sides=sides,
        left_can=str(cfg_get(l6_cfg, "left_can", "can0")),
        right_can=str(cfg_get(l6_cfg, "right_can", "can1")),
        modbus=str(cfg_get(l6_cfg, "modbus", "None")),
        rate_hz=_positive_float(cfg_get(hands_cfg, "rate_hz", cfg_get(l6_cfg, "rate_hz", 30.0)), "rate_hz"),
        frame_timeout_s=_positive_float(cfg_get(hands_cfg, "frame_timeout_s", 0.3), "frame_timeout_s"),
        trigger_deadzone=_deadzone(cfg_get(l6_cfg, "trigger_deadzone", 0.05)),
        deadman_threshold=_threshold(cfg_get(l6_cfg, "deadman_threshold", 0.5)),
        thumb_yaw_center=thumb_yaw,
        speed=tuple(speed),
        open_pose=tuple(open_pose),
        close_pose=tuple(close_pose),
        fixed_thumb_yaw=thumb_yaw,
        print_input=bool(cfg_get(l6_cfg, "print_input", False)),
        somehand_config_path=str(cfg_get(somehand_cfg, "config_path", DEFAULT_SOMEHAND_CONFIG)),
        somehand_rate_hz=_positive_float(cfg_get(somehand_cfg, "rate_hz", cfg_get(somehand_cfg, "rate", 60.0)), "somehand.rate_hz"),
        somehand_max_iterations=_optional_positive_int(cfg_get(somehand_cfg, "max_iterations", None), "somehand.max_iterations"),
        somehand_temporal_filter_alpha=_optional_alpha(cfg_get(somehand_cfg, "temporal_filter_alpha", None), "somehand.temporal_filter_alpha"),
        somehand_output_alpha=_optional_alpha(cfg_get(somehand_cfg, "output_alpha", None), "somehand.output_alpha"),
    )


class LinkerHandL6Device(HandDevice):
    def __init__(self, config: LinkerHandL6Config):
        self.config = config
        self._hands: dict[str, Any] = {}
        self._last_pose: dict[str, tuple[int, ...] | None] = {side: None for side in config.sides}

    def connect(self) -> None:
        try:
            from LinkerHand.linker_hand_api import LinkerHandApi
        except ImportError as exc:
            raise ImportError(
                "LinkerHand SDK is required for hands.driver=linkerhand_l6. "
                "Install it with: pip install -e third_party/linkerhand-python-sdk"
            ) from exc
        try:
            for side in self.config.sides:
                hand = LinkerHandApi(
                    hand_joint="L6",
                    hand_type=side,
                    modbus=self.config.modbus,
                    can=self.config.left_can if side == "left" else self.config.right_can,
                )
                hand.set_speed(speed=list(self.config.speed))
                self._hands[side] = hand
        except (Exception, SystemExit) as exc:
            self.close()
            if isinstance(exc, SystemExit):
                raise RuntimeError("LinkerHand SDK exited during startup") from exc
            raise
        self.open_all(force=True, reason="startup")

    def send_pose(self, side: str, pose: Sequence[int], *, force: bool = False, reason: str = "") -> None:
        del reason
        next_pose = tuple(_uint8(value, f"{side}.pose") for value in pose)
        if not force and self._last_pose.get(side) == next_pose:
            return
        hand = self._hands.get(side)
        if hand is None:
            return
        hand.finger_move(pose=list(next_pose))
        self._last_pose[side] = next_pose

    def open_all(self, *, force: bool = False, reason: str = "") -> None:
        for side in self.config.sides:
            self.send_pose(side, self.config.open_pose, force=force, reason=reason)

    def close(self) -> None:
        try:
            self.open_all(force=True, reason="shutdown")
        except Exception:
            logger.exception("Failed to open LinkerHand L6 on shutdown")
        for hand in self._hands.values():
            inner = getattr(hand, "hand", None)
            close = getattr(inner, "close", None)
            if callable(close):
                close()
        self._hands.clear()


class GripperMapper(HandInputMapper):
    def __init__(self, config: Any):
        self.config = config
        self._fixed_thumb_yaw = getattr(config, "fixed_thumb_yaw", getattr(config, "thumb_yaw_center", None))
        self._active = False
        self._next_tick_s = 0.0

    def start(self) -> None:
        pass

    def map(self, *, controller_snapshot: object | None, hand_snapshot: object | None, active: bool, now_s: float) -> tuple[HandPoseCommand, ...]:
        del hand_snapshot
        if not active:
            if not self._active:
                return ()
            self._active = False
            return tuple(HandPoseCommand(side, self.config.open_pose, True, "inactive") for side in self.config.sides)
        if now_s < self._next_tick_s:
            return ()
        self._active = True
        self._next_tick_s = now_s + 1.0 / self.config.rate_hz
        if controller_snapshot is None or now_s - float(getattr(controller_snapshot, "timestamp_s", 0.0)) > self.config.frame_timeout_s:
            return tuple(HandPoseCommand(side, self.config.open_pose, False, "timeout") for side in self.config.sides)
        commands: list[HandPoseCommand] = []
        for side in self.config.sides:
            state = getattr(controller_snapshot, side)
            if not bool(getattr(state, "present", True)):
                commands.append(HandPoseCommand(side, self.config.open_pose, False, "missing-controller"))
                continue
            grip = _clamp01(getattr(state, "grip", 0.0))
            trigger = _clamp01(getattr(state, "trigger", 0.0))
            if grip < self.config.deadman_threshold:
                commands.append(HandPoseCommand(side, self.config.open_pose, False, "deadman"))
                continue
            pose = trigger_to_pose(
                trigger,
                open_pose=self.config.open_pose,
                close_pose=self.config.close_pose,
                deadzone=self.config.trigger_deadzone,
                fixed_thumb_yaw=self._fixed_thumb_yaw,
            )
            commands.append(HandPoseCommand(side, tuple(pose), False, "controller"))
        return tuple(commands)

    def close(self) -> None:
        pass


class SomehandL6Mapper(HandInputMapper):
    def __init__(self, config: LinkerHandL6Config):
        self.config = config
        self._engine: Any | None = None
        self._hand_frame_cls: Any | None = None
        self._bihand_frame_cls: Any | None = None
        self._mappers: dict[str, L6RetargetPoseMapper] = {}
        self._next_tick_s = 0.0
        self._active = False

    def start(self) -> None:
        _require_somehand_020()
        from somehand.api import HandFrame, RetargetingEngine, load_bihand_config, load_retargeting_config

        config_path = _resolve_project_path(self.config.somehand_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"somehand L6 config not found: {config_path}")
        bihand_config = load_bihand_config(str(config_path))
        self._engine = {}
        for side, path in (("left", bihand_config.left_config_path), ("right", bihand_config.right_config_path)):
            retarget_cfg = load_retargeting_config(path)
            self._apply_low_latency_overrides(retarget_cfg)
            self._engine[side] = RetargetingEngine(retarget_cfg)
        self._hand_frame_cls = HandFrame
        for side, engine in self._engine.items():
            if side in self.config.sides:
                self._mappers[side] = L6RetargetPoseMapper(getattr(engine, "hand_model", None), side=side)

    def map(self, *, controller_snapshot: object | None, hand_snapshot: object | None, active: bool, now_s: float) -> tuple[HandPoseCommand, ...]:
        del controller_snapshot
        if not active:
            if not self._active:
                return ()
            self._active = False
            return tuple(HandPoseCommand(side, self.config.open_pose, True, "inactive") for side in self.config.sides)
        if now_s < self._next_tick_s:
            return ()
        self._active = True
        self._next_tick_s = now_s + 1.0 / self.config.somehand_rate_hz
        if hand_snapshot is None or now_s - float(getattr(hand_snapshot, "timestamp_s", 0.0)) > self.config.frame_timeout_s:
            return ()
        left_frame = self._make_frame("left", getattr(hand_snapshot, "left", None))
        right_frame = self._make_frame("right", getattr(hand_snapshot, "right", None))
        if left_frame is None and right_frame is None:
            return ()
        commands: list[HandPoseCommand] = []
        for side, frame in (("left", left_frame), ("right", right_frame)):
            if side not in self.config.sides or frame is None:
                continue
            step = self._engine[side].process(frame)
            commands.append(HandPoseCommand(side, tuple(self._mappers[side].qpos_to_pose(step.qpos)), False, "vr-hand-pose"))
        return tuple(commands)

    def close(self) -> None:
        pass

    def _make_frame(self, side: str, state: object | None) -> object | None:
        if side not in self.config.sides or state is None:
            return None
        if not bool(getattr(state, "present", False)) or not bool(getattr(state, "active", False)):
            return None
        landmarks = pico_hand_to_landmarks(getattr(state, "joints"))
        return self._hand_frame_cls(landmarks_3d=landmarks, landmarks_2d=None, hand_side=side)

    def _apply_low_latency_overrides(self, cfg: object) -> None:
        if self.config.somehand_max_iterations is not None:
            cfg.solver.max_iterations = int(self.config.somehand_max_iterations)
        if self.config.somehand_output_alpha is not None:
            cfg.solver.output_alpha = float(self.config.somehand_output_alpha)
        if self.config.somehand_temporal_filter_alpha is not None:
            cfg.preprocess.temporal_filter_alpha = float(self.config.somehand_temporal_filter_alpha)


class L6RetargetPoseMapper:
    def __init__(self, hand_model: Any | None, *, side: str):
        if hand_model is None:
            raise ValueError("somehand L6 hand model is missing")
        get_index = getattr(hand_model, "get_joint_name_to_qpos_index", None)
        if not callable(get_index):
            raise ValueError("somehand L6 hand model does not expose get_joint_name_to_qpos_index()")
        joint_index = get_index()
        self._indices = np.asarray([_resolve_l6_joint_index(joint_index, name, side=side) for name in L6_SDK_JOINT_ORDER], dtype=np.int64)
        mapping = _load_linkerhand_mapping_module()
        side_key = "l" if side == "left" else "r"
        self._mapping = mapping
        self._arc_min = np.asarray(getattr(mapping, f"l6_{side_key}_min"), dtype=np.float64)
        self._arc_max = np.asarray(getattr(mapping, f"l6_{side_key}_max"), dtype=np.float64)
        self._direction = np.asarray(getattr(mapping, f"l6_{side_key}_derict"), dtype=np.int8)

    def qpos_to_pose(self, qpos: object) -> list[int]:
        values = np.asarray(qpos, dtype=np.float64).reshape(-1)
        selected = values[self._indices]
        pose = []
        for index, value in enumerate(selected):
            arc = self._mapping.is_within_range(float(value), float(self._arc_min[index]), float(self._arc_max[index]))
            if int(self._direction[index]) == -1:
                scaled = self._mapping.scale_value(arc, float(self._arc_min[index]), float(self._arc_max[index]), 255.0, 0.0)
            else:
                scaled = self._mapping.scale_value(arc, float(self._arc_min[index]), float(self._arc_max[index]), 0.0, 255.0)
            pose.append(_uint8(round(float(scaled)), "somehand.pose"))
        return pose


def build_linkerhand_l6(cfg: Any) -> tuple[HandDevice, HandInputMapper]:
    config = parse_linkerhand_l6_config(cfg)
    device = LinkerHandL6Device(config)
    mapper: HandInputMapper = SomehandL6Mapper(config) if config.mode == "vr_hand_pose" else GripperMapper(config)
    return device, mapper


def trigger_to_pose(
    trigger: float,
    *,
    open_pose: Sequence[int],
    close_pose: Sequence[int],
    deadzone: float,
    fixed_thumb_yaw: int | None = None,
    thumb_yaw_default: int | None = None,
) -> list[int]:
    if fixed_thumb_yaw is None:
        fixed_thumb_yaw = thumb_yaw_default
    alpha = _normalize_trigger(trigger, deadzone)
    pose = [int(round(float(a) + alpha * (float(b) - float(a)))) for a, b in zip(open_pose, close_pose)]
    if fixed_thumb_yaw is not None:
        pose[1] = int(fixed_thumb_yaw)
    return pose


def _require_somehand_020() -> None:
    try:
        installed = version("somehand")
    except PackageNotFoundError as exc:
        raise ImportError("somehand==0.2.0 is required for hands.mode=vr_hand_pose") from exc
    if installed != "0.2.0":
        raise ImportError(f"somehand==0.2.0 is required for hands.mode=vr_hand_pose, found {installed}")


def _resolve_l6_joint_index(joint_index: dict[str, int], semantic_name: str, *, side: str) -> int:
    for candidate in _l6_joint_candidates(semantic_name, side=side):
        if candidate in joint_index:
            return int(joint_index[candidate])
    suffixes = tuple(f"_{alias}" for alias in _l6_aliases(semantic_name))
    for name, index in joint_index.items():
        if name in _l6_aliases(semantic_name) or any(name.endswith(suffix) for suffix in suffixes):
            return int(index)
    raise ValueError(f"Cannot resolve LinkerHand L6 SDK joint {semantic_name!r} in somehand hand model")


def _l6_joint_candidates(semantic_name: str, *, side: str) -> tuple[str, ...]:
    prefixes = ("", f"{side}_", f"{side[0]}_", f"{side[0].upper()}_", f"{'lh' if side == 'left' else 'rh'}_")
    return tuple(f"{prefix}{alias}" for alias in _l6_aliases(semantic_name) for prefix in prefixes)


def _l6_aliases(semantic_name: str) -> tuple[str, ...]:
    if semantic_name == "thumb_cmc_pitch":
        return ("thumb_cmc_pitch", "thumb_pitch")
    if semantic_name == "thumb_cmc_roll":
        return ("thumb_cmc_roll", "thumb_roll")
    aliases = [semantic_name]
    if semantic_name.endswith("_mcp_pitch"):
        finger = semantic_name[: -len("_mcp_pitch")]
        aliases.append(f"{finger}_pitch")
        if finger == "pinky":
            aliases.extend(("little_mcp_pitch", "little_pitch"))
    return tuple(dict.fromkeys(aliases))


def _load_linkerhand_mapping_module() -> Any:
    mapping_path = _resolve_project_path(DEFAULT_LINKERHAND_SDK_ROOT) / "LinkerHand" / "utils" / "mapping.py"
    spec = importlib.util.spec_from_file_location("teleopit_linkerhand_mapping", mapping_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LinkerHand mapping module from {mapping_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _clamp01(value: object) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_trigger(value: float, deadzone: float) -> float:
    value = _clamp01(value)
    if value <= deadzone:
        return 0.0
    upper = 1.0 - deadzone
    if value >= upper:
        return 1.0
    return (value - deadzone) / (upper - deadzone)


def _uint8(value: object, field_name: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > 255:
        raise ValueError(f"hands.linkerhand_l6.{field_name} must be in 0-255, got {value!r}")
    return parsed


def _pose_values(value: object, field_name: str) -> list[int]:
    parsed = [_uint8(item, field_name) for item in value]  # type: ignore[union-attr]
    if len(parsed) != 6:
        raise ValueError(f"hands.linkerhand_l6.{field_name} must contain 6 values")
    return parsed


def _positive_float(value: object, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"hands.{field_name} must be > 0")
    return parsed


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"hands.{field_name} must be > 0")
    return parsed


def _optional_alpha(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"hands.{field_name} must be in (0, 1]")
    return parsed


def _deadzone(value: object) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed >= 0.5:
        raise ValueError("hands.linkerhand_l6.trigger_deadzone must be in [0, 0.5)")
    return parsed


def _threshold(value: object) -> float:
    parsed = float(value)
    if parsed <= 0.0 or parsed >= 1.0:
        raise ValueError("hands.linkerhand_l6.deadman_threshold must be in (0, 1)")
    return parsed
