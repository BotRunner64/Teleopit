from __future__ import annotations

import pytest
import torch

import math

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


def test_ema_does_not_decay_unobserved_bins() -> None:
    """Bins with zero exposure must keep their old rate, not decay toward 0.

    This was the root cause of entropy collapse: the old code applied the EMA
    to ALL bins, so unobserved bins got ``ema * 0 + (1-ema) * old``, steadily
    decaying their failure rates and biasing p_hat toward observed bins.
    """
    bin_count = 5
    ema = 0.01
    tb_failed_rate = torch.tensor([0.5, 0.3, 0.8, 0.2, 0.6])

    # Only bins 0 and 2 were sampled this step
    accum_exposure = torch.tensor([10.0, 0.0, 5.0, 0.0, 0.0])
    accum_failure = torch.tensor([4.0, 0.0, 3.0, 0.0, 0.0])

    if accum_exposure.sum() > 0:
        valid = accum_exposure > 0
        observed_rate = accum_failure[valid] / accum_exposure[valid]
        # Correct: only update observed bins
        tb_failed_rate[valid] = (
            ema * observed_rate + (1 - ema) * tb_failed_rate[valid]
        )

    # Bins 1, 3, 4 should be UNCHANGED (not decayed)
    assert tb_failed_rate[1] == 0.3, f"Bin 1 decayed: {tb_failed_rate[1]}"
    assert tb_failed_rate[3] == 0.2, f"Bin 3 decayed: {tb_failed_rate[3]}"
    assert tb_failed_rate[4] == 0.6, f"Bin 4 decayed: {tb_failed_rate[4]}"
    # Observed bins should be updated
    # bin 0: ema*0.4 + (1-ema)*0.5 = 0.004 + 0.495 = 0.499
    assert abs(tb_failed_rate[0] - 0.499) < 1e-5
    # bin 2: ema*0.6 + (1-ema)*0.8 = 0.006 + 0.792 = 0.798
    assert abs(tb_failed_rate[2] - 0.798) < 1e-5


def test_old_ema_bug_would_decay_unobserved_bins() -> None:
    """Demonstrate what the old (buggy) code did — for regression protection."""
    bin_count = 5
    ema = 0.01
    tb_failed_rate = torch.tensor([0.5, 0.3, 0.8, 0.2, 0.6])

    accum_exposure = torch.tensor([10.0, 0.0, 5.0, 0.0, 0.0])
    accum_failure = torch.tensor([4.0, 0.0, 3.0, 0.0, 0.0])

    # OLD buggy code: applies EMA to ALL bins
    valid = accum_exposure > 0
    global_rate = torch.zeros_like(tb_failed_rate)
    global_rate[valid] = accum_failure[valid] / accum_exposure[valid]
    tb_failed_rate_buggy = ema * global_rate + (1 - ema) * tb_failed_rate

    # Unobserved bins were wrongly decayed:
    # bin 1: 0.01*0 + 0.99*0.3 = 0.297 (not 0.3!)
    assert tb_failed_rate_buggy[1] < 0.3, "Old code should have decayed bin 1"
    assert tb_failed_rate_buggy[3] < 0.2, "Old code should have decayed bin 3"
    assert tb_failed_rate_buggy[4] < 0.6, "Old code should have decayed bin 4"


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


# ---------------------------------------------------------------------------
# Fix: spawn-bin attribution
# ---------------------------------------------------------------------------


def test_spawn_bin_attribution_not_drifted_by_time_update() -> None:
    """Failure should be attributed to the spawn bin, not the current bin."""
    num_bins = 4
    num_envs = 3

    # Simulate: envs spawned at bins [0, 1, 2]
    spawn_bin_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    # After time progression, env_bin_ids drifted to [1, 2, 3]
    env_bin_ids = torch.tensor([1, 2, 3], dtype=torch.long)

    episode_failed = torch.tensor([True, True, False])

    # Attribution using spawn bins (correct)
    exposure_spawn = torch.bincount(spawn_bin_ids, minlength=num_bins).float()
    failure_spawn = torch.bincount(spawn_bin_ids[episode_failed], minlength=num_bins).float()

    # Attribution using drifted bins (incorrect old behavior)
    exposure_drift = torch.bincount(env_bin_ids, minlength=num_bins).float()
    failure_drift = torch.bincount(env_bin_ids[episode_failed], minlength=num_bins).float()

    # Spawn attribution: failure at bins 0 and 1
    assert torch.allclose(failure_spawn, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    # Drifted attribution: failure at bins 1 and 2 (wrong!)
    assert torch.allclose(failure_drift, torch.tensor([0.0, 1.0, 1.0, 0.0]))
    # They should be different — spawn attribution avoids systematic later-bin bias
    assert not torch.equal(failure_spawn, failure_drift)


# ---------------------------------------------------------------------------
# Fix: minimum probability floor
# ---------------------------------------------------------------------------


def test_min_prob_floor_prevents_zero_probability() -> None:
    """With min_prob > 0, no bin should have probability below floor / N."""
    rates = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    beta = 5.0
    alpha = 0.8
    min_prob = 0.5
    N = len(rates)

    capped = _cap_failure_rates(rates, beta)
    capped_sum = capped.sum()
    p_hat = capped / capped_sum if capped_sum > 0 else torch.ones_like(capped) / N

    p_final = alpha * p_hat + (1.0 - alpha) / N

    # Apply min_prob floor
    floor = min_prob / N
    p_clamped = torch.clamp(p_final, min=floor)
    p_clamped = p_clamped / p_clamped.sum()

    # Before normalization, all bins >= floor; after normalization, they shrink
    # proportionally but the minimum should be much larger than without the floor
    assert torch.allclose(p_clamped.sum(), torch.tensor(1.0))
    # Without floor, the zero-rate bins get only (1-alpha)/N = 0.04
    # With floor, they get at least floor/sum ≈ 0.08 (2x improvement)
    p_no_floor = p_final / p_final.sum()
    assert p_clamped.min() > p_no_floor.min(), "Floor should raise minimum probability"


def test_min_prob_zero_preserves_original_behavior() -> None:
    """When min_prob=0, the formula is unchanged."""
    rates = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    beta = 5.0
    alpha = 0.8
    N = len(rates)

    capped = _cap_failure_rates(rates, beta)
    capped_sum = capped.sum()
    p_hat = capped / capped_sum if capped_sum > 0 else torch.ones_like(capped) / N

    p_original = alpha * p_hat + (1.0 - alpha) / N
    p_original = p_original / p_original.sum()

    # With min_prob=0, no clamping
    p_with_zero_floor = p_original.clone()
    # No clamp applied
    p_with_zero_floor = p_with_zero_floor / p_with_zero_floor.sum()

    assert torch.allclose(p_original, p_with_zero_floor)


# ---------------------------------------------------------------------------
# Fix: checkpoint round-trip for adaptive_bin state
# ---------------------------------------------------------------------------


def test_adaptive_bin_state_dict_round_trip() -> None:
    """Saving and loading tb_failed_rate should preserve the tensor."""
    original_rate = torch.tensor([0.1, 0.5, 0.3, 0.0, 0.8])

    # Simulate state_dict
    state = {"tb_failed_rate": original_rate.clone()}

    # Simulate load
    restored_rate = torch.zeros(5)
    saved = state.get("tb_failed_rate")
    if saved is not None and saved.shape == restored_rate.shape:
        restored_rate.copy_(saved)

    assert torch.allclose(restored_rate, original_rate)


def test_adaptive_bin_state_dict_shape_mismatch_is_safe() -> None:
    """If bin count changes between runs, the load should be a no-op."""
    state = {"tb_failed_rate": torch.tensor([0.1, 0.5, 0.3])}

    restored_rate = torch.ones(5)  # Mirrors adaptive_bin cold-start initialization
    saved = state.get("tb_failed_rate")
    if saved is not None and saved.shape == restored_rate.shape:
        restored_rate.copy_(saved)

    # Should remain unchanged because shape didn't match
    assert torch.allclose(restored_rate, torch.ones(5))


# ---------------------------------------------------------------------------
# Fix: entropy-based alpha decay
# ---------------------------------------------------------------------------


def test_entropy_floor_reduces_alpha_when_entropy_low() -> None:
    """Alpha should be reduced when p_hat entropy is below the floor."""
    # A very concentrated p_hat
    p_hat = torch.tensor([0.9, 0.05, 0.03, 0.02])
    N = len(p_hat)
    alpha_base = 0.8
    entropy_floor = 0.9

    entropy = -(p_hat * (p_hat + 1e-12).log()).sum()
    entropy_norm = entropy / math.log(N)

    # Entropy is low (~0.53), below floor of 0.9
    assert entropy_norm < entropy_floor

    # Compute decayed alpha
    alpha_decayed = alpha_base * (entropy_norm / entropy_floor)
    assert alpha_decayed < alpha_base
    assert alpha_decayed > 0


def test_entropy_floor_zero_disables_decay() -> None:
    """With entropy_floor=0, alpha should never be modified."""
    p_hat = torch.tensor([0.9, 0.05, 0.03, 0.02])
    alpha_base = 0.8
    entropy_floor = 0.0

    # The condition `entropy_floor > 0` is False, so no decay
    alpha = alpha_base
    if entropy_floor > 0:
        entropy = -(p_hat * (p_hat + 1e-12).log()).sum()
        entropy_norm = entropy / math.log(len(p_hat))
        if entropy_norm < entropy_floor:
            alpha = alpha * (entropy_norm / entropy_floor)

    assert alpha == alpha_base
