from __future__ import annotations

import torch

from train_mimic.tasks.tracking.mdp.commands import (
    _compute_clip_failure_counts,
    _compute_clip_failure_rate,
    _compute_failure_rate_from_counts,
    _compute_rank_sample_range,
    _compute_windowed_ema_alpha,
)


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


def test_compute_clip_failure_counts_and_rate_round_trip() -> None:
    motion_ids = torch.tensor([0, 2, 2, 2, 4, 4], dtype=torch.long)
    episode_failed = torch.tensor([False, True, False, True, False, False])

    exposure_count, failure_count = _compute_clip_failure_counts(
        motion_ids, episode_failed, bin_count=6
    )
    rate = _compute_failure_rate_from_counts(exposure_count, failure_count)

    assert torch.equal(exposure_count, torch.tensor([1.0, 0.0, 3.0, 0.0, 2.0, 0.0]))
    assert torch.equal(failure_count, torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(
        rate, torch.tensor([0.0, 0.0, 2.0 / 3.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    )


def test_compute_rank_sample_range_splits_global_batch() -> None:
    request_counts = torch.tensor([3, 0, 2, 4], dtype=torch.long)

    assert _compute_rank_sample_range(request_counts, 0) == (0, 3)
    assert _compute_rank_sample_range(request_counts, 1) == (3, 3)
    assert _compute_rank_sample_range(request_counts, 2) == (3, 5)
    assert _compute_rank_sample_range(request_counts, 3) == (5, 9)


def test_compute_windowed_ema_alpha_matches_repeated_single_step_updates() -> None:
    alpha = 0.001
    num_steps = 24

    expected = 1.0 - (1.0 - alpha) ** num_steps

    assert _compute_windowed_ema_alpha(alpha, num_steps) == expected
