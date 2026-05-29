from __future__ import annotations

import numpy as np

from teleopit.inputs.human_frame_validation import validate_human_frame


def test_validate_human_frame_reports_out_of_range_joint() -> None:
    frame = {
        "Pelvis": (np.array([6.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
    }

    result = validate_human_frame(frame, max_pos_value=5.0)

    assert not result.valid
    assert result.reason == "position_out_of_range"
    assert result.joint_name == "Pelvis"
    assert result.pos == (6.0, 0.0, 0.0)
    assert result.max_abs_pos == 6.0
    assert result.max_pos_value == 5.0


def test_validate_human_frame_reports_quaternion_nan() -> None:
    frame = {
        "Head": (np.array([0.0, 0.0, 1.5]), np.array([1.0, np.nan, 0.0, 0.0])),
    }

    result = validate_human_frame(frame, max_pos_value=5.0)

    assert not result.valid
    assert result.reason == "quaternion_nan"
    assert result.joint_name == "Head"


def test_validate_human_frame_accepts_finite_frame() -> None:
    frame = {
        "Pelvis": (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        "Head": (np.array([0.0, 0.0, 1.7]), np.array([1.0, 0.0, 0.0, 0.0])),
    }

    result = validate_human_frame(frame, max_pos_value=5.0)

    assert result.valid
    assert result.reason == "ok"
