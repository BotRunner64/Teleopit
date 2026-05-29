"""HumanFrame finite-value validation shared by realtime diagnostics and runtimes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HumanFrameValidationResult:
    valid: bool
    reason: str = "ok"
    joint_name: str | None = None
    pos: tuple[float, ...] | None = None
    quat: tuple[float, ...] | None = None
    max_abs_pos: float | None = None
    detail: str = ""


def validate_human_frame(frame: object) -> HumanFrameValidationResult:
    """Validate HumanFrame numeric values before realtime retargeting."""
    if not isinstance(frame, dict):
        return HumanFrameValidationResult(False, reason="frame_not_dict", detail=f"type={type(frame)!r}")

    max_frame_abs_pos: float | None = None
    for name, value in frame.items():
        joint_name = str(name)
        try:
            pos, quat = value
        except Exception as exc:
            return HumanFrameValidationResult(
                False,
                reason="joint_unpack_failed",
                joint_name=joint_name,
                detail=str(exc),
            )

        try:
            pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        except Exception as exc:
            return HumanFrameValidationResult(
                False,
                reason="position_cast_failed",
                joint_name=joint_name,
                detail=str(exc),
            )
        try:
            quat_arr = np.asarray(quat, dtype=np.float64).reshape(-1)
        except Exception as exc:
            return HumanFrameValidationResult(
                False,
                reason="quaternion_cast_failed",
                joint_name=joint_name,
                pos=_to_tuple(pos_arr),
                detail=str(exc),
            )

        pos_tuple = _to_tuple(pos_arr)
        quat_tuple = _to_tuple(quat_arr)
        max_abs_pos = float(np.max(np.abs(pos_arr))) if pos_arr.size > 0 else 0.0
        max_frame_abs_pos = (
            max_abs_pos if max_frame_abs_pos is None else max(max_frame_abs_pos, max_abs_pos)
        )

        if np.any(np.isnan(pos_arr)):
            return HumanFrameValidationResult(
                False,
                reason="position_nan",
                joint_name=joint_name,
                pos=pos_tuple,
                quat=quat_tuple,
                max_abs_pos=max_abs_pos,
            )
        if np.any(np.isinf(pos_arr)):
            return HumanFrameValidationResult(
                False,
                reason="position_inf",
                joint_name=joint_name,
                pos=pos_tuple,
                quat=quat_tuple,
                max_abs_pos=max_abs_pos,
            )
        if np.any(np.isnan(quat_arr)):
            return HumanFrameValidationResult(
                False,
                reason="quaternion_nan",
                joint_name=joint_name,
                pos=pos_tuple,
                quat=quat_tuple,
                max_abs_pos=max_abs_pos,
            )
        if np.any(np.isinf(quat_arr)):
            return HumanFrameValidationResult(
                False,
                reason="quaternion_inf",
                joint_name=joint_name,
                pos=pos_tuple,
                quat=quat_tuple,
                max_abs_pos=max_abs_pos,
            )

    return HumanFrameValidationResult(True, max_abs_pos=max_frame_abs_pos)


def _to_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1))
