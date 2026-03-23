"""Registry wiring for supported tracking tasks."""

from mjlab.tasks.registry import register_mjlab_task

from train_mimic.tasks.tracking.config.constants import (
    MOTION_TRACKING_DEPLOY_EXPERIMENT_NAME,
    MOTION_TRACKING_DEPLOY_TASK,
    VELCMD_HISTORY_ADAPTIVE_EXPERIMENT_NAME,
    VELCMD_HISTORY_ADAPTIVE_TASK,
    VELCMD_HISTORY_REGULAR_EXPERIMENT_NAME,
    VELCMD_HISTORY_REGULAR_TASK,
    VELCMD_HISTORY_EXPERIMENT_NAME,
    VELCMD_HISTORY_TASK,
    VELCMD_REF_WINDOW_EXPERIMENT_NAME,
    VELCMD_REF_WINDOW_TASK,
)
from train_mimic.tasks.tracking.config.env import (
    make_motion_tracking_deploy_env_cfg,
    make_velcmd_history_adaptive_tracking_env_cfg,
    make_velcmd_history_regular_tracking_env_cfg,
    make_velcmd_history_tracking_env_cfg,
    make_velcmd_ref_window_tracking_env_cfg,
)
from train_mimic.tasks.tracking.config.rl import (
    make_motion_tracking_deploy_ppo_runner_cfg,
    make_velcmd_history_regular_tracking_ppo_runner_cfg,
    make_velcmd_history_tracking_ppo_runner_cfg,
    make_velcmd_ref_window_tracking_ppo_runner_cfg,
)
from train_mimic.tasks.tracking.rl import MotionTrackingOnPolicyRunner

register_mjlab_task(
    task_id=VELCMD_HISTORY_TASK,
    env_cfg=make_velcmd_history_tracking_env_cfg(),
    play_env_cfg=make_velcmd_history_tracking_env_cfg(play=True),
    rl_cfg=make_velcmd_history_tracking_ppo_runner_cfg(
        experiment_name=VELCMD_HISTORY_EXPERIMENT_NAME
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id=VELCMD_HISTORY_ADAPTIVE_TASK,
    env_cfg=make_velcmd_history_adaptive_tracking_env_cfg(),
    play_env_cfg=make_velcmd_history_adaptive_tracking_env_cfg(play=True),
    rl_cfg=make_velcmd_history_tracking_ppo_runner_cfg(
        experiment_name=VELCMD_HISTORY_ADAPTIVE_EXPERIMENT_NAME
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id=VELCMD_REF_WINDOW_TASK,
    env_cfg=make_velcmd_ref_window_tracking_env_cfg(),
    play_env_cfg=make_velcmd_ref_window_tracking_env_cfg(play=True),
    rl_cfg=make_velcmd_ref_window_tracking_ppo_runner_cfg(
        experiment_name=VELCMD_REF_WINDOW_EXPERIMENT_NAME
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id=MOTION_TRACKING_DEPLOY_TASK,
    env_cfg=make_motion_tracking_deploy_env_cfg(),
    play_env_cfg=make_motion_tracking_deploy_env_cfg(play=True),
    rl_cfg=make_motion_tracking_deploy_ppo_runner_cfg(
        experiment_name=MOTION_TRACKING_DEPLOY_EXPERIMENT_NAME
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id=VELCMD_HISTORY_REGULAR_TASK,
    env_cfg=make_velcmd_history_regular_tracking_env_cfg(),
    play_env_cfg=make_velcmd_history_regular_tracking_env_cfg(play=True),
    rl_cfg=make_velcmd_history_regular_tracking_ppo_runner_cfg(
        experiment_name=VELCMD_HISTORY_REGULAR_EXPERIMENT_NAME
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)
