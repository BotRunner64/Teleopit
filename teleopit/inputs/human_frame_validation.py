"""HumanFrame sanity validation shared by realtime diagnostics and runtimes."""

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
    max_pos_value: float | None = None
    detail: str = ""


def validate_human_frame(frame: object, *, max_pos_value: float) -> HumanFrameValidationResult:
    """Validate the same HumanFrame conditions used before realtime retargeting."""
    if not isinstance(frame, dict):
        return HumanFrameValidationResult(False, reason="frame_not_dict", detail=f"type={type(frame)!r}")

    max_pos = float(max_pos_value)
    if not np.isfinite(max_pos) or max_pos <= 0.0:
        return HumanFrameValidationResult(
            False,
            reason="invalid_max_position_value",
            max_pos_value=max_pos,
        )

    for name, value in frame.items():
        joint_name = str(name)
        try:
            pos, quat = value
        except Exception as exc:
            return HumanFrameValidationResult(
                False,
                reason="joint_unpack_failed",
                joint_name=joint_name,
                max_pos_value=max_pos,
                detail=str(exc),
            )

        try:
            pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        except Exception as exc:
            return HumanFrameValidationResult(
                False,
                reason="position_cast_failed",
                joint_name=joint_name,
                max_pos_value=max_pos,
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
                max_pos_value=max_pos,
                detail=str(exc),
            )

        pos_tuple = _to_tuple(pos_arr)
        quat_tuple = _to_tuple(quat_arr)
        max_abs_pos = float(np.max(np.abs(pos_arr))) if pos_arr.size > 0 else 0.0

        if np.any(np.isnan(pos_arr)):
            return HumanFrameValidationResult(
                False,
                reason="position_nan",
                joint_name=joint_name,
                pos=pos_tuple,
                quat=quat_tuple,
                max_abs_pos=max_abs_pos,
                max_pos_value=max_pos,
            )
        if np.any(np.isinf(pos_arr)):
            return HumanFrameValidationResult(
                False,
                reason="position_inf",
                joint_name=joint_name,
                pos=pos_tuple,
                quat=quat_tuple,
                max_abs_pos=max_abs_pos,
                max_pos_value=max_pos,
            )
        if np.any(np.abs(pos_arr) > max_pos):
            return HumanFrameValidationResult(
                False,
                reason="position_out_of_range",
                joint_name=joint_name,
                pos=pos_tuple,
                quat=quat_tuple,
                max_abs_pos=max_abs_pos,
                max_pos_value=max_pos,
            )
        if np.any(np.isnan(quat_arr)):
            return HumanFrameValidationResult(
                False,
                reason="quaternion_nan",
                joint_name=joint_name,
                pos=pos_tuple,
                quat=quat_tuple,
                max_abs_pos=max_abs_pos,
                max_pos_value=max_pos,
            )
        if np.any(np.isinf(quat_arr)):
            return HumanFrameValidationResult(
                False,
                reason="quaternion_inf",
                joint_name=joint_name,
                pos=pos_tuple,
                quat=quat_tuple,
                max_abs_pos=max_abs_pos,
                max_pos_value=max_pos,
            )

    return HumanFrameValidationResult(True, max_pos_value=max_pos)


def _to_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1))
