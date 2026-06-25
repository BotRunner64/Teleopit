from __future__ import annotations

import numpy as np

from teleopit.inputs.human_frame_validation import validate_human_frame


def test_validate_human_frame_accepts_large_finite_positions() -> None:
    frame = {
        "Pelvis": (np.array([100.0, -100.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        "Head": (np.array([100.1, -100.1, 1.7]), np.array([1.0, 0.0, 0.0, 0.0])),
    }

    result = validate_human_frame(frame)

    assert result.valid
    assert result.reason == "ok"


def test_validate_human_frame_reports_position_inf() -> None:
    frame = {
        "Pelvis": (np.array([np.inf, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
    }

    result = validate_human_frame(frame)

    assert not result.valid
    assert result.reason == "position_inf"
    assert result.joint_name == "Pelvis"


def test_validate_human_frame_reports_position_nan() -> None:
    frame = {
        "Pelvis": (np.array([np.nan, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
    }

    result = validate_human_frame(frame)

    assert not result.valid
    assert result.reason == "position_nan"
    assert result.joint_name == "Pelvis"


def test_validate_human_frame_reports_quaternion_nan() -> None:
    frame = {
        "Head": (np.array([0.0, 0.0, 1.5]), np.array([1.0, np.nan, 0.0, 0.0])),
    }

    result = validate_human_frame(frame)

    assert not result.valid
    assert result.reason == "quaternion_nan"
    assert result.joint_name == "Head"


def test_validate_human_frame_accepts_finite_frame() -> None:
    frame = {
        "Pelvis": (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        "Head": (np.array([0.0, 0.0, 1.7]), np.array([1.0, 0.0, 0.0, 0.0])),
    }

    result = validate_human_frame(frame)

    assert result.valid
    assert result.reason == "ok"
