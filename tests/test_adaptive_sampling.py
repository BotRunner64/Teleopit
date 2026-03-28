from __future__ import annotations

import pytest
import torch

from train_mimic.tasks.tracking.mdp.commands import (
    _cap_failure_rates,
    _compute_clip_counts,
    _compute_clip_failure_rate,
    _normalize_sampling_probabilities,
    _validate_legacy_adaptive_config,
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


def test_validate_legacy_adaptive_config_rejects_nondefault_kernel_size() -> None:
    with pytest.raises(ValueError, match="adaptive_kernel_size"):
        _validate_legacy_adaptive_config(adaptive_kernel_size=3, adaptive_lambda=0.8)


def test_validate_legacy_adaptive_config_rejects_nondefault_lambda() -> None:
    with pytest.raises(ValueError, match="adaptive_lambda"):
        _validate_legacy_adaptive_config(adaptive_kernel_size=1, adaptive_lambda=0.5)


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


# ---------------------------------------------------------------------------
# adaptive_bin: _cap_failure_rates
# ---------------------------------------------------------------------------


def test_cap_failure_rates_clamps_outliers() -> None:
    rates = torch.tensor([0.1, 0.9, 0.2])
    # mean = 0.4, beta=2.0 -> cap = 0.8
    capped = _cap_failure_rates(rates, beta=2.0)
    expected = torch.tensor([0.1, 0.8, 0.2])
    assert torch.allclose(capped, expected)


def test_cap_failure_rates_all_zero() -> None:
    rates = torch.zeros(5)
    capped = _cap_failure_rates(rates, beta=5.0)
    assert torch.allclose(capped, torch.zeros(5))


def test_cap_failure_rates_uniform() -> None:
    """When all rates are equal, capping changes nothing."""
    rates = torch.tensor([0.3, 0.3, 0.3])
    capped = _cap_failure_rates(rates, beta=2.0)
    assert torch.allclose(capped, rates)


# ---------------------------------------------------------------------------
# adaptive_bin: build_time_bins (via MotionLib mock)
# ---------------------------------------------------------------------------


def _make_mock_motion_lib(
    clip_sample_start_s: list[float],
    clip_sample_end_s: list[float],
    clip_weights: list[float],
):
    """Create a lightweight mock with just the fields build_time_bins needs."""
    import math as _math
    from types import SimpleNamespace

    from train_mimic.tasks.tracking.mdp.commands import MotionLib

    num_clips = len(clip_weights)
    device = "cpu"

    mock = SimpleNamespace()
    mock.num_clips = num_clips
    mock._device = device
    mock.clip_weights = torch.tensor(clip_weights, dtype=torch.float32, device=device)
    mock.clip_sample_start_s = torch.tensor(
        clip_sample_start_s, dtype=torch.float32, device=device
    )
    mock.clip_sample_end_s = torch.tensor(
        clip_sample_end_s, dtype=torch.float32, device=device
    )
    # Bind the real method
    mock.build_time_bins = MotionLib.build_time_bins.__get__(mock, type(mock))
    return mock


def test_build_time_bins_single_clip() -> None:
    ml = _make_mock_motion_lib(
        clip_sample_start_s=[0.0],
        clip_sample_end_s=[12.0],
        clip_weights=[1.0],
    )
    result = ml.build_time_bins(bin_duration_s=5.0)

    assert result["num_bins"] == 3  # [0,5), [5,10), [10,12)
    assert torch.equal(result["bin_clip_id"], torch.tensor([0, 0, 0]))
    assert torch.allclose(result["bin_start_s"], torch.tensor([0.0, 5.0, 10.0]))
    assert torch.allclose(result["bin_end_s"], torch.tensor([5.0, 10.0, 12.0]))
    assert torch.allclose(result["bin_duration"], torch.tensor([5.0, 5.0, 2.0]))
    assert result["clip_bin_offset"][0].item() == 0


def test_build_time_bins_multiple_clips() -> None:
    ml = _make_mock_motion_lib(
        clip_sample_start_s=[0.0, 0.0, 0.0],
        clip_sample_end_s=[3.0, 7.0, 10.0],
        clip_weights=[1.0, 1.0, 1.0],
    )
    result = ml.build_time_bins(bin_duration_s=5.0)

    # clip 0: 3s -> 1 bin; clip 1: 7s -> 2 bins; clip 2: 10s -> 2 bins
    assert result["num_bins"] == 5
    expected_clip_ids = torch.tensor([0, 1, 1, 2, 2])
    assert torch.equal(result["bin_clip_id"], expected_clip_ids)


def test_build_time_bins_skips_zero_weight() -> None:
    ml = _make_mock_motion_lib(
        clip_sample_start_s=[0.0, 0.0],
        clip_sample_end_s=[10.0, 10.0],
        clip_weights=[1.0, 0.0],
    )
    result = ml.build_time_bins(bin_duration_s=5.0)

    assert result["num_bins"] == 2  # only clip 0
    assert torch.equal(result["bin_clip_id"], torch.tensor([0, 0]))
    assert result["clip_bin_offset"][1].item() == -1


def test_build_time_bins_short_clip() -> None:
    """A clip shorter than bin_duration produces exactly 1 bin."""
    ml = _make_mock_motion_lib(
        clip_sample_start_s=[0.0],
        clip_sample_end_s=[2.0],
        clip_weights=[1.0],
    )
    result = ml.build_time_bins(bin_duration_s=5.0)

    assert result["num_bins"] == 1
    assert torch.allclose(result["bin_duration"], torch.tensor([2.0]))


# ---------------------------------------------------------------------------
# adaptive_bin: probability formula
# ---------------------------------------------------------------------------


def test_adaptive_bin_probability_formula() -> None:
    """Verify the full capped + blended probability computation."""
    rates = torch.tensor([0.1, 0.9, 0.2, 0.0])
    beta = 2.0
    alpha = 0.8
    N = len(rates)

    # Step 1: cap
    capped = _cap_failure_rates(rates, beta)
    # mean=0.3, cap=0.6 -> [0.1, 0.6, 0.2, 0.0]
    expected_capped = torch.tensor([0.1, 0.6, 0.2, 0.0])
    assert torch.allclose(capped, expected_capped)

    # Step 2: normalize to p_hat
    capped_sum = capped.sum()
    p_hat = capped / capped_sum  # [0.1/0.9, 0.6/0.9, 0.2/0.9, 0]

    # Step 3: blend
    p_final = alpha * p_hat + (1.0 - alpha) / N
    p_final = p_final / p_final.sum()

    # Verify it's a valid distribution
    assert torch.allclose(p_final.sum(), torch.tensor(1.0))
    assert (p_final > 0).all(), "All bins should have positive probability"
    # Bin 1 (highest failure) should have highest probability
    assert p_final[1] == p_final.max()
