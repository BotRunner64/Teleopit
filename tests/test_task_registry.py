"""Tests for supported tracking task registration."""

from __future__ import annotations

import pytest

from train_mimic.app import DEFAULT_TASK
from train_mimic.tasks.tracking.config.constants import (
    VELCMD_HISTORY_ADAPTIVE_EXPERIMENT_NAME,
    VELCMD_HISTORY_ADAPTIVE_TASK,
    VELCMD_HISTORY_EXPERIMENT_NAME,
    VELCMD_HISTORY_REGULAR_EXPERIMENT_NAME,
    VELCMD_HISTORY_REGULAR_TASK,
    VELCMD_HISTORY_TASK,
    VELCMD_REF_WINDOW_EXPERIMENT_NAME,
    VELCMD_REF_WINDOW_TASK,
)
from train_mimic.tasks.tracking.rl import MotionTrackingOnPolicyRunner


def test_velcmd_history_task_is_registered() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(DEFAULT_TASK)
    actor_terms = env_cfg.observations["actor"].terms
    critic_terms = env_cfg.observations["critic"].terms

    assert DEFAULT_TASK == VELCMD_HISTORY_TASK
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
    assert load_rl_cfg(DEFAULT_TASK).experiment_name == VELCMD_HISTORY_EXPERIMENT_NAME
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


def test_velcmd_history_adaptive_task_is_registered() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(VELCMD_HISTORY_ADAPTIVE_TASK)

    assert env_cfg.commands["motion"].sampling_mode == "adaptive"
    assert env_cfg.commands["motion"].window_steps == (0,)
    assert load_rl_cfg(VELCMD_HISTORY_ADAPTIVE_TASK).experiment_name == (
        VELCMD_HISTORY_ADAPTIVE_EXPERIMENT_NAME
    )
    assert load_runner_cls(VELCMD_HISTORY_ADAPTIVE_TASK) is MotionTrackingOnPolicyRunner


def test_velcmd_ref_window_task_is_registered() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(VELCMD_REF_WINDOW_TASK)
    rl_cfg = load_rl_cfg(VELCMD_REF_WINDOW_TASK)

    assert env_cfg.commands["motion"].sampling_mode == "uniform"
    assert env_cfg.commands["motion"].window_steps == (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10
    )
    assert "actor_history" in env_cfg.observations
    assert "critic_history" in env_cfg.observations
    assert "actor_ref_window" in env_cfg.observations
    assert "critic_ref_window" in env_cfg.observations
    assert rl_cfg.obs_groups["actor"] == ("actor", "actor_history", "actor_ref_window")
    assert rl_cfg.experiment_name == VELCMD_REF_WINDOW_EXPERIMENT_NAME
    assert load_runner_cls(VELCMD_REF_WINDOW_TASK) is MotionTrackingOnPolicyRunner


def test_velcmd_history_regular_task_is_registered() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(VELCMD_HISTORY_REGULAR_TASK)

    assert env_cfg.commands["motion"].feet_body_names == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert "joint_pos_tracking" in env_cfg.rewards
    assert "joint_vel_tracking" in env_cfg.rewards
    assert "feet_air_time" in env_cfg.rewards
    assert "feet_stumble" in env_cfg.rewards
    assert "feet_contact_forces" in env_cfg.rewards
    assert "feet_slip" in env_cfg.rewards
    assert "joint_torque_limits" in env_cfg.rewards
    assert load_rl_cfg(VELCMD_HISTORY_REGULAR_TASK).experiment_name == (
        VELCMD_HISTORY_REGULAR_EXPERIMENT_NAME
    )
    assert load_runner_cls(VELCMD_HISTORY_REGULAR_TASK) is MotionTrackingOnPolicyRunner


def test_removed_task_variants_are_not_registered() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    with pytest.raises(Exception):
        load_env_cfg("Tracking-Flat-G1-LegacyRemoved")
