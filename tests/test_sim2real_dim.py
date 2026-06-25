from __future__ import annotations

from types import SimpleNamespace

import pytest

from teleopit.sim2real.mp.runtime import _RobotControlWorker


def _cfg() -> dict[str, object]:
    return {
        "policy_hz": 50.0,
        "input": {"provider": "bvh"},
        "runtime": {},
        "real_robot": {},
        "controller": {},
        "robot": {
            "num_actions": 29,
            "default_angles": [0.0] * 29,
            "xml_path": "robot.xml",
        },
    }


def test_robot_worker_requires_dual_input_policy(monkeypatch) -> None:
    policy = SimpleNamespace(_multi_input=False)
    obs_builder = SimpleNamespace(total_obs_size=167)
    monkeypatch.setattr(
        "teleopit.sim2real.mp.runtime._build_policy_components",
        lambda **_kwargs: (policy, obs_builder),
    )

    worker = object.__new__(_RobotControlWorker)
    worker.cfg = _cfg()

    with pytest.raises(ValueError, match="dual inputs"):
        worker._build_policy_and_obs()


def test_robot_worker_accepts_167d_dual_input_policy(monkeypatch) -> None:
    policy = SimpleNamespace(_multi_input=True)
    obs_builder = SimpleNamespace(total_obs_size=167)
    monkeypatch.setattr(
        "teleopit.sim2real.mp.runtime._build_policy_components",
        lambda **_kwargs: (policy, obs_builder),
    )

    worker = object.__new__(_RobotControlWorker)
    worker.cfg = _cfg()

    built_policy, built_obs_builder = worker._build_policy_and_obs()

    assert built_policy is policy
    assert built_obs_builder is obs_builder
