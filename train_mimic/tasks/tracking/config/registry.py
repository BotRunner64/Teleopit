"""Registry wiring for the single supported tracking task."""

from mjlab.tasks.registry import register_mjlab_task

from train_mimic.tasks.tracking.config.constants import (
    VELCMD_HISTORY_EXPERIMENT_NAME,
    VELCMD_HISTORY_TASK,
)
from train_mimic.tasks.tracking.config.env import make_velcmd_history_tracking_env_cfg
from train_mimic.tasks.tracking.config.rl import make_velcmd_history_tracking_ppo_runner_cfg
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
