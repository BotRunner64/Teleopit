"""Tests for supported tracking task registration."""

from __future__ import annotations

import pytest

from train_mimic.app import DEFAULT_TASK
from train_mimic.tasks.tracking.config.constants import (
    GENERAL_TRACKING_EXPERIMENT_NAME,
    GENERAL_TRACKING_TASK,
)
from train_mimic.tasks.tracking.rl import MotionTrackingOnPolicyRunner


def test_general_tracking_task_is_registered() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(DEFAULT_TASK)
    actor_terms = env_cfg.observations["actor"].terms
    critic_terms = env_cfg.observations["critic"].terms

    assert DEFAULT_TASK == GENERAL_TRACKING_TASK
    for terms in (actor_terms, critic_terms):
        assert "projected_gravity" in terms
        assert "ref_base_lin_vel_b" in terms
        assert "ref_base_ang_vel_b" in terms
        assert "ref_projected_gravity_b" in terms
    assert "motion_anchor_pos_b" not in actor_terms
    assert "base_lin_vel" not in actor_terms
    assert "actor_history" in env_cfg.observations
    assert "critic_history" in env_cfg.observations
    assert env_cfg.commands["motion"].sampling_mode == "uniform"
    assert env_cfg.commands["motion"].window_steps == (0,)
    rl_cfg = load_rl_cfg(DEFAULT_TASK)
    assert rl_cfg.experiment_name == GENERAL_TRACKING_EXPERIMENT_NAME
    assert rl_cfg.actor.hidden_dims == (1024, 512, 256, 256, 128)
    assert load_runner_cls(DEFAULT_TASK) is MotionTrackingOnPolicyRunner


def test_play_env_disables_corruption_and_random_push() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    play_cfg = load_env_cfg(DEFAULT_TASK, play=True)

    assert play_cfg.observations["actor"].enable_corruption is False
    assert play_cfg.observations["actor_history"].enable_corruption is False
    assert play_cfg.observations["critic_history"].enable_corruption is False
    assert "push_robot" not in play_cfg.events
    assert play_cfg.commands["motion"].sampling_mode == "start"
    assert play_cfg.commands["motion"].window_steps == (0,)


def test_removed_task_variants_are_not_registered() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    removed_tasks = [
        "Tracking-Flat-G1-VelCmdHistory",
        "Tracking-Flat-G1-VelCmdHistoryAdaptive",
        "Tracking-Flat-G1-VelCmdHistoryRegular",
        "Tracking-Flat-G1-VelCmdHistoryLarge",
        "Tracking-Flat-G1-VelCmdRefWindow",
        "Tracking-Flat-G1-MotionTrackingDeploy",
        "Tracking-Flat-G1-LegacyRemoved",
    ]
    for task in removed_tasks:
        with pytest.raises(Exception):
            load_env_cfg(task)
