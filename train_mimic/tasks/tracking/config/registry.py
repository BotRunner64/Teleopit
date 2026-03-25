"""Registry wiring for the General-Tracking-G1 task."""

from mjlab.tasks.registry import register_mjlab_task

from train_mimic.tasks.tracking.config.constants import (
    GENERAL_TRACKING_EXPERIMENT_NAME,
    GENERAL_TRACKING_TASK,
)
from train_mimic.tasks.tracking.config.env import make_general_tracking_env_cfg
from train_mimic.tasks.tracking.config.rl import make_general_tracking_ppo_runner_cfg
from train_mimic.tasks.tracking.rl import MotionTrackingOnPolicyRunner

register_mjlab_task(
    task_id=GENERAL_TRACKING_TASK,
    env_cfg=make_general_tracking_env_cfg(),
    play_env_cfg=make_general_tracking_env_cfg(play=True),
    rl_cfg=make_general_tracking_ppo_runner_cfg(
        experiment_name=GENERAL_TRACKING_EXPERIMENT_NAME
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)
