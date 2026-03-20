from __future__ import annotations

import numpy as np

from teleopit.sim.reference_timeline import ReferenceTimeline, ReferenceWindowBuilder
from teleopit.sim.realtime_utils import ExponentialVecSmoother, RealtimeReferenceManager


def _qpos(x: float) -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float64)
    qpos[0] = x
    qpos[3] = 1.0
    return qpos


def test_exponential_vec_smoother_blends_and_resets() -> None:
    smoother = ExponentialVecSmoother(alpha=0.25)

    first = smoother.apply(np.array([0.0, 0.0], dtype=np.float32))
    second = smoother.apply(np.array([4.0, 8.0], dtype=np.float32))

    np.testing.assert_allclose(first, np.array([0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(second, np.array([1.0, 2.0], dtype=np.float32), atol=1e-6)

    smoother.reset()
    reset_value = smoother.apply(np.array([4.0, 8.0], dtype=np.float32))
    np.testing.assert_allclose(reset_value, np.array([4.0, 8.0], dtype=np.float32), atol=1e-6)


def test_realtime_reference_manager_warmup_counts_real_frames() -> None:
    manager = RealtimeReferenceManager(
        reference_window_builder=ReferenceWindowBuilder(policy_dt_s=0.02, reference_steps=[0, 1, 2]),
        low_watermark_steps=1,
        high_watermark_steps=3,
        warmup_steps=2,
    )

    assert manager.warmup_done is False
    manager.note_realtime_frame()
    assert manager.warmup_done is False
    manager.note_realtime_frame()
    assert manager.warmup_done is True


def test_realtime_reference_manager_repeat_pads_future_window() -> None:
    timeline = ReferenceTimeline(window_s=1.0)
    timeline.append(_qpos(0.0), 0.0)
    timeline.append(_qpos(1.0), 0.02)

    manager = RealtimeReferenceManager(
        reference_window_builder=ReferenceWindowBuilder(policy_dt_s=0.02, reference_steps=[0, 1, 2]),
        low_watermark_steps=1,
        high_watermark_steps=2,
        warmup_steps=0,
    )
    manager.note_realtime_frame()
    manager.note_realtime_frame()

    window, diagnostics = manager.sample(timeline, 0.02)

    assert diagnostics.future_horizon_steps == 0
    assert diagnostics.used_repeat_padding is True
    assert diagnostics.padding_active is True
    np.testing.assert_allclose(window.current_sample().qpos[0], 1.0, atol=1e-6)
    assert window.samples[1].mode == 'repeat_latest'
    assert window.samples[2].mode == 'repeat_latest'
    np.testing.assert_allclose(window.samples[1].timestamp_s, 0.04, atol=1e-6)
    np.testing.assert_allclose(window.samples[2].timestamp_s, 0.06, atol=1e-6)


def test_realtime_reference_manager_defaults_high_watermark_to_effective_low() -> None:
    manager = RealtimeReferenceManager(
        reference_window_builder=ReferenceWindowBuilder(policy_dt_s=0.02, reference_steps=[0, 1, 2, 3, 4]),
        low_watermark_steps=0,
        high_watermark_steps=None,
        warmup_steps=0,
    )

    assert manager._low_watermark_steps == 4
    assert manager._high_watermark_steps == 4
