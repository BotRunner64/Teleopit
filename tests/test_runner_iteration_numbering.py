from __future__ import annotations

import pytest

from train_mimic.tasks.tracking.rl.runner import (
    _one_based_iteration_range,
    _resolve_total_iterations,
)


def test_one_based_iteration_range_starts_at_one_for_fresh_run() -> None:
    assert list(_one_based_iteration_range(0, 10)) == list(range(1, 11))


def test_one_based_iteration_range_resumes_from_completed_iteration() -> None:
    assert list(_one_based_iteration_range(10, 12)) == [11, 12]


def test_one_based_iteration_range_is_empty_when_already_at_target() -> None:
    assert list(_one_based_iteration_range(10, 10)) == []


def test_one_based_iteration_range_rejects_target_below_completed() -> None:
    with pytest.raises(ValueError, match='num_learning_iterations'):
        _one_based_iteration_range(11, 10)


def test_resolve_total_iterations_preserves_fresh_run_count() -> None:
    assert _resolve_total_iterations(0, 10) == 10


def test_resolve_total_iterations_adds_requested_iterations_on_resume() -> None:
    assert _resolve_total_iterations(10, 12) == 22


def test_resolve_total_iterations_rejects_negative_requested_iterations() -> None:
    with pytest.raises(ValueError, match='non-negative'):
        _resolve_total_iterations(10, -1)
