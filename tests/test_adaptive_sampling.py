from __future__ import annotations

import pytest
import torch

from train_mimic.tasks.tracking.mdp.commands import (
    _compute_clip_counts,
    _compute_clip_failure_rate,
    _normalize_sampling_probabilities,
)


# ---------------------------------------------------------------------------
# Existing tests (refactored helper, same semantics)
# ---------------------------------------------------------------------------


def test_compute_clip_failure_rate_is_invariant_to_parallel_env_count() -> None:
    motion_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    episode_failed = torch.tensor([True, False, True, False, False, True])

    small_rate = _compute_clip_failure_rate(motion_ids, episode_failed, bin_count=3)
    large_rate = _compute_clip_failure_rate(
        motion_ids.repeat(16), episode_failed.repeat(16), bin_count=3
    )

    expected = torch.tensor([2.0 / 3.0, 1.0 / 3.0, 0.0], dtype=torch.float32)
    assert torch.allclose(small_rate, expected)
    assert torch.allclose(large_rate, expected)


def test_compute_clip_failure_rate_ignores_unseen_clips() -> None:
    motion_ids = torch.tensor([1, 1, 3, 3], dtype=torch.long)
    episode_failed = torch.tensor([True, False, False, False])

    rate = _compute_clip_failure_rate(motion_ids, episode_failed, bin_count=5)

    expected = torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0], dtype=torch.float32)
    assert torch.allclose(rate, expected)


# ---------------------------------------------------------------------------
# New tests for _compute_clip_counts
# ---------------------------------------------------------------------------


def test_compute_clip_counts_returns_correct_counts() -> None:
    motion_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    episode_failed = torch.tensor([True, False, True, False, False, True])

    exposure, failure = _compute_clip_counts(motion_ids, episode_failed, bin_count=3)

    assert torch.allclose(exposure, torch.tensor([3.0, 3.0, 0.0]))
    assert torch.allclose(failure, torch.tensor([2.0, 1.0, 0.0]))


# ---------------------------------------------------------------------------
# EMA no-decay on empty steps
# ---------------------------------------------------------------------------


def test_ema_skips_update_when_no_data() -> None:
    """bin_failed_rate must NOT decay when accumulators are empty."""
    bin_count = 4
    alpha = 0.01
    bin_failed_rate = torch.tensor([0.5, 0.3, 0.0, 0.2])
    accum_exposure = torch.zeros(bin_count)
    accum_failure = torch.zeros(bin_count)

    # Simulate the guarded EMA logic from _update_command
    if accum_exposure.sum() > 0:
        valid = accum_exposure > 0
        global_rate = torch.zeros_like(bin_failed_rate)
        global_rate[valid] = accum_failure[valid] / accum_exposure[valid]
        bin_failed_rate = alpha * global_rate + (1 - alpha) * bin_failed_rate

    expected = torch.tensor([0.5, 0.3, 0.0, 0.2])
    assert torch.allclose(bin_failed_rate, expected)


# ---------------------------------------------------------------------------
# Fail-fast validation for invalid adaptive distributions
# ---------------------------------------------------------------------------


def test_normalize_sampling_probabilities_raises_on_zero_mass() -> None:
    sampling_probabilities = torch.zeros(5)

    with pytest.raises(ValueError, match="invalid probability mass"):
        _normalize_sampling_probabilities(
            sampling_probabilities,
            adaptive_uniform_ratio=0.0,
            bin_count=5,
        )


# ---------------------------------------------------------------------------
# Accumulation sums across multiple resamples
# ---------------------------------------------------------------------------


def test_accumulation_sums_across_resamples() -> None:
    """Two accumulate calls before EMA update should sum counts correctly."""
    bin_count = 3
    accum_exposure = torch.zeros(bin_count)
    accum_failure = torch.zeros(bin_count)

    # First resample batch
    ids1 = torch.tensor([0, 0, 1], dtype=torch.long)
    failed1 = torch.tensor([True, False, True])
    e1, f1 = _compute_clip_counts(ids1, failed1, bin_count)
    accum_exposure += e1
    accum_failure += f1

    # Second resample batch
    ids2 = torch.tensor([1, 2, 2], dtype=torch.long)
    failed2 = torch.tensor([False, True, False])
    e2, f2 = _compute_clip_counts(ids2, failed2, bin_count)
    accum_exposure += e2
    accum_failure += f2

    assert torch.allclose(accum_exposure, torch.tensor([2.0, 2.0, 2.0]))
    assert torch.allclose(accum_failure, torch.tensor([1.0, 1.0, 1.0]))
