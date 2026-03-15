"""Tests for mjlab task registration used by train_mimic."""

from __future__ import annotations

from train_mimic.app import DEFAULT_TASK
from train_mimic.tasks.tracking.rl import MotionTrackingOnPolicyRunner


def test_official_no_state_estimation_task_is_registered() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(DEFAULT_TASK)
    actor_terms = env_cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in actor_terms
    assert "base_lin_vel" not in actor_terms
    assert "motion_anchor_ori_b" in actor_terms
    assert "base_ang_vel" in actor_terms
    assert env_cfg.commands["motion"].sampling_mode == "uniform"

    assert load_rl_cfg(DEFAULT_TASK).experiment_name == "g1_tracking"
    assert load_runner_cls(DEFAULT_TASK) is MotionTrackingOnPolicyRunner


def test_play_env_disables_corruption_and_random_push() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    play_cfg = load_env_cfg(DEFAULT_TASK, play=True)
    actor_terms = play_cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in actor_terms
    assert "base_lin_vel" not in actor_terms
    assert play_cfg.observations["actor"].enable_corruption is False
    assert "push_robot" not in play_cfg.events
    assert play_cfg.commands["motion"].sampling_mode == "start"
