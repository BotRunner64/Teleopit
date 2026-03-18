from __future__ import annotations

import torch

from train_mimic.tasks.tracking.mdp.commands import _compute_clip_failure_rate


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
