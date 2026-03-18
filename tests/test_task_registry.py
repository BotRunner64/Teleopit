"""Tests for mjlab task registration used by train_mimic."""

from __future__ import annotations

from train_mimic.app import DEFAULT_TASK
from train_mimic.tasks.tracking.config.constants import (
    VELCMD_HISTORY_ADAPTIVE_TASK,
    VELCMD_HISTORY_ADAPTIVE_EXPERIMENT_NAME,
    VELCMD_HISTORY_TASK,
    VELCMD_HISTORY_EXPERIMENT_NAME,
)
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


def test_velcmd_history_adaptive_task_is_registered() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(VELCMD_HISTORY_ADAPTIVE_TASK)
    play_cfg = load_env_cfg(VELCMD_HISTORY_ADAPTIVE_TASK, play=True)
    actor_terms = env_cfg.observations["actor"].terms
    critic_terms = env_cfg.observations["critic"].terms

    for terms in (actor_terms, critic_terms):
        assert "projected_gravity" in terms
        assert "ref_base_lin_vel_b" in terms
        assert "ref_base_ang_vel_b" in terms
        assert "ref_projected_gravity_b" in terms

    assert "actor_history" in env_cfg.observations
    assert "critic_history" in env_cfg.observations
    assert env_cfg.commands["motion"].sampling_mode == "adaptive"
    assert env_cfg.commands["motion"].adaptive_sync_steps == 24
    assert env_cfg.commands["motion"].adaptive_log_interval == 50
    assert play_cfg.commands["motion"].sampling_mode == "start"
    assert play_cfg.observations["actor_history"].enable_corruption is False
    assert play_cfg.observations["critic_history"].enable_corruption is False
    assert load_rl_cfg(VELCMD_HISTORY_ADAPTIVE_TASK).experiment_name == (
        VELCMD_HISTORY_ADAPTIVE_EXPERIMENT_NAME
    )
    assert load_runner_cls(VELCMD_HISTORY_ADAPTIVE_TASK) is MotionTrackingOnPolicyRunner


def test_velcmd_history_adaptive_matches_uniform_variant_except_sampling_and_experiment() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    uniform_env_cfg = load_env_cfg(VELCMD_HISTORY_TASK)
    adaptive_env_cfg = load_env_cfg(VELCMD_HISTORY_ADAPTIVE_TASK)

    assert tuple(uniform_env_cfg.observations["actor"].terms) == tuple(
        adaptive_env_cfg.observations["actor"].terms
    )
    assert tuple(uniform_env_cfg.observations["critic"].terms) == tuple(
        adaptive_env_cfg.observations["critic"].terms
    )
    assert tuple(uniform_env_cfg.observations["actor_history"].terms) == tuple(
        adaptive_env_cfg.observations["actor_history"].terms
    )
    assert tuple(uniform_env_cfg.observations["critic_history"].terms) == tuple(
        adaptive_env_cfg.observations["critic_history"].terms
    )
    assert uniform_env_cfg.commands["motion"].sampling_mode == "uniform"
    assert adaptive_env_cfg.commands["motion"].sampling_mode == "adaptive"
    assert load_rl_cfg(VELCMD_HISTORY_TASK).experiment_name == VELCMD_HISTORY_EXPERIMENT_NAME
    assert load_rl_cfg(VELCMD_HISTORY_ADAPTIVE_TASK).experiment_name == (
        VELCMD_HISTORY_ADAPTIVE_EXPERIMENT_NAME
    )
