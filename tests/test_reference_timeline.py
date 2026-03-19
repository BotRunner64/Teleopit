from __future__ import annotations

import numpy as np
import pytest

from teleopit.sim.reference_timeline import ReferenceTimeline, ReferenceWindowBuilder, parse_reference_steps


def _qpos(x: float, yaw_deg: float = 0.0) -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float64)
    qpos[0] = x
    angle = np.deg2rad(yaw_deg)
    qpos[3:7] = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)], dtype=np.float64)
    return qpos


def test_parse_reference_steps_defaults_to_current() -> None:
    assert parse_reference_steps(None) == (0,)


def test_parse_reference_steps_rejects_missing_zero() -> None:
    with pytest.raises(ValueError, match="contain 0"):
        parse_reference_steps([1, 2, -1])


def test_parse_reference_steps_rejects_future_after_history() -> None:
    with pytest.raises(ValueError, match="future/non-negative"):
        parse_reference_steps([0, -1, 1])


def test_reference_timeline_interpolates_between_frames() -> None:
    timeline = ReferenceTimeline(window_s=1.0)
    timeline.append(_qpos(0.0, yaw_deg=0.0), 1.0)
    timeline.append(_qpos(1.0, yaw_deg=90.0), 2.0)

    sample = timeline.sample_at(1.5)

    assert sample.mode == "interpolate"
    assert sample.used_fallback is False
    np.testing.assert_allclose(sample.qpos[0], 0.5, atol=1e-6)
    np.testing.assert_allclose(sample.qpos[3:7], _qpos(0.0, yaw_deg=45.0)[3:7], atol=1e-6)


def test_reference_timeline_falls_back_at_edges() -> None:
    timeline = ReferenceTimeline(window_s=1.0)
    timeline.append(_qpos(0.0), 1.0)
    timeline.append(_qpos(1.0), 2.0)

    oldest = timeline.sample_at(0.5)
    latest = timeline.sample_at(2.5)

    assert oldest.mode == "fallback_oldest"
    assert latest.mode == "fallback_latest"
    np.testing.assert_allclose(oldest.qpos[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(latest.qpos[0], 1.0, atol=1e-6)


def test_reference_timeline_collapses_duplicate_timestamps() -> None:
    timeline = ReferenceTimeline(window_s=1.0)
    timeline.append(_qpos(0.0), 1.0)
    timeline.append(_qpos(1.0), 1.0)

    assert len(timeline) == 1
    sample = timeline.sample_at(1.0)
    assert sample.mode == "single_frame"
    np.testing.assert_allclose(sample.qpos[0], 1.0, atol=1e-6)


def test_reference_timeline_trims_by_window() -> None:
    timeline = ReferenceTimeline(window_s=0.5)
    timeline.append(_qpos(0.0), 1.0)
    timeline.append(_qpos(1.0), 1.4)
    timeline.append(_qpos(2.0), 2.0)

    assert len(timeline) == 1
    sample = timeline.sample_at(2.0)
    np.testing.assert_allclose(sample.qpos[0], 2.0, atol=1e-6)


def test_reference_window_builder_samples_future_and_history() -> None:
    timeline = ReferenceTimeline(window_s=5.0)
    for idx in range(6):
        timeline.append(_qpos(float(idx)), float(idx))

    builder = ReferenceWindowBuilder(policy_dt_s=1.0, reference_steps=[0, 1, 2, -1, -2])
    window = builder.sample(timeline, 3.0)

    assert window.reference_steps == (0, 1, 2, -1, -2)
    np.testing.assert_allclose(
        np.asarray([sample.qpos[0] for sample in window.samples], dtype=np.float64),
        np.array([3.0, 4.0, 5.0, 2.0, 1.0], dtype=np.float64),
        atol=1e-6,
    )


def test_reference_window_builder_validates_future_delay_requirement() -> None:
    builder = ReferenceWindowBuilder(policy_dt_s=0.02, reference_steps=[0, 2])

    with pytest.raises(ValueError, match="retarget_buffer_delay_s"):
        builder.validate_runtime_support(delay_s=0.03, window_s=1.0, config_label="test")


def test_reference_window_builder_validates_window_requirement() -> None:
    builder = ReferenceWindowBuilder(policy_dt_s=0.02, reference_steps=[0, -2])

    with pytest.raises(ValueError, match="retarget_buffer_window_s"):
        builder.validate_runtime_support(delay_s=0.03, window_s=0.06, config_label="test")
