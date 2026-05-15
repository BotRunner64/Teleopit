from __future__ import annotations

from types import SimpleNamespace

import torch

from train_mimic.tasks.tracking.mdp.rewards import self_collision_cost


def _env_with_force_history(force_history: torch.Tensor) -> SimpleNamespace:
    sensor = SimpleNamespace(
        data=SimpleNamespace(force_history=force_history, found=None)
    )
    return SimpleNamespace(scene={"self_collision": sensor})


def test_self_collision_cost_counts_contact_slots_not_history_frames() -> None:
    force_history = torch.zeros((2, 3, 4, 3), dtype=torch.float32)
    force_history[0, 0, 0, 0] = 2.0
    force_history[0, 1, 0, 0] = 3.0
    force_history[0, 2, 2, 2] = 2.0
    force_history[1, 0, 1, 0] = 0.5
    force_history[1, 0, 3, 0] = 1.0

    penalty = self_collision_cost(
        _env_with_force_history(force_history),
        sensor_name="self_collision",
        force_threshold=1.0,
    )

    torch.testing.assert_close(penalty, torch.tensor([3.0, 0.0]))
