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
    assert list(actor_terms) == [
        "ref_joint_pos",
        "ref_joint_vel",
        "ref_anchor_ori_b",
        "robot_base_ang_vel_b",
        "robot_joint_pos_rel",
        "robot_joint_vel",
        "prev_action",
        "robot_projected_gravity_b",
        "ref_anchor_lin_vel_b",
        "ref_anchor_ang_vel_b",
        "ref_projected_gravity_b",
        "ref_anchor_height",
    ]
    assert list(critic_terms) == [
        "ref_joint_pos",
        "ref_joint_vel",
        "ref_anchor_pos_b",
        "ref_anchor_ori_b",
        "robot_tracking_body_pos_b",
        "robot_tracking_body_ori_b",
        "robot_base_lin_vel_b",
        "robot_base_ang_vel_b",
        "robot_joint_pos_rel",
        "robot_joint_vel",
        "prev_action",
        "robot_projected_gravity_b",
        "ref_anchor_lin_vel_b",
        "ref_anchor_ang_vel_b",
        "ref_projected_gravity_b",
        "ref_anchor_height",
    ]
    assert "actor_history" in env_cfg.observations
    assert "critic_history" in env_cfg.observations
    assert env_cfg.commands["motion"].sampling_mode == "rewind"
    assert env_cfg.commands["motion"].window_steps == (0,)
    assert env_cfg.rewards["motion_global_root_lin_vel"].weight == 1.0
    assert env_cfg.rewards["motion_global_root_lin_vel"].params == {
        "command_name": "motion",
        "std": 1.0,
    }
    assert env_cfg.rewards["motion_global_root_ang_vel"].weight == 1.0
    assert env_cfg.rewards["motion_global_root_ang_vel"].params == {
        "command_name": "motion",
        "std": 3.0,
    }
    assert env_cfg.rewards["motion_joint_pos"].weight == 1.0
    assert env_cfg.rewards["motion_joint_pos"].params == {
        "command_name": "motion",
        "std": 0.5,
    }
    assert env_cfg.rewards["motion_joint_vel"].weight == 0.5
    assert env_cfg.rewards["motion_joint_vel"].params == {
        "command_name": "motion",
        "std": 3.0,
    }
    assert env_cfg.rewards["survival"].weight == 3.0
    assert env_cfg.rewards["survival"].params == {}
    reward = env_cfg.rewards["self_collisions"]
    assert reward.weight == -0.1
    assert reward.params == {
        "sensor_name": "self_collision",
        "force_threshold": 1.0,
    }
    assert "undesired_contacts" not in env_cfg.rewards
    sensors = {sensor.name: sensor for sensor in env_cfg.scene.sensors}
    assert set(sensors) == {"self_collision"}
    assert sensors["self_collision"].primary.mode == "body"
    assert sensors["self_collision"].primary.pattern == r".*"
    assert sensors["self_collision"].primary.exclude == (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    assert sensors["self_collision"].secondary.mode == "subtree"
    assert sensors["self_collision"].secondary.pattern == "pelvis"
    assert sensors["self_collision"].reduce == "maxforce"
    assert sensors["self_collision"].history_length == 4
    feet_acc = env_cfg.rewards["feet_acc"]
    assert feet_acc.weight == -2.5e-6
    assert feet_acc.params["asset_cfg"].name == "robot"
    assert feet_acc.params["asset_cfg"].joint_names == r".*ankle.*"
    assert "anti_shake_ang_vel" not in env_cfg.rewards
    rl_cfg = load_rl_cfg(DEFAULT_TASK)
    assert rl_cfg.experiment_name == GENERAL_TRACKING_EXPERIMENT_NAME
    assert rl_cfg.actor.hidden_dims == (2048, 1024, 512, 256, 128)
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
