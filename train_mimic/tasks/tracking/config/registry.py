"""Registry wiring for the official tracking task surface."""

from mjlab.tasks.registry import register_mjlab_task

from train_mimic.tasks.tracking.config.constants import (
    HISTORY_CNN_TASK,
    NO_ROOT_POSE_EXPERIMENT_NAME,
    NO_ROOT_POSE_TASK,
    OFFICIAL_TASK,
)
from train_mimic.tasks.tracking.config.env import (
    make_history_cnn_tracking_env_cfg,
    make_tracking_env_cfg_for_profile,
)
from train_mimic.tasks.tracking.config.profiles import OFFICIAL_UNIFORM_PROFILE
from train_mimic.tasks.tracking.config.rl import (
    make_history_cnn_tracking_ppo_runner_cfg,
    make_tracking_ppo_runner_cfg,
)
from train_mimic.tasks.tracking.rl import MotionTrackingOnPolicyRunner

register_mjlab_task(
    task_id=OFFICIAL_TASK,
    env_cfg=make_tracking_env_cfg_for_profile(OFFICIAL_UNIFORM_PROFILE),
    play_env_cfg=make_tracking_env_cfg_for_profile(OFFICIAL_UNIFORM_PROFILE, play=True),
    rl_cfg=make_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id=NO_ROOT_POSE_TASK,
    env_cfg=make_tracking_env_cfg_for_profile(
        OFFICIAL_UNIFORM_PROFILE, no_root_pose=True
    ),
    play_env_cfg=make_tracking_env_cfg_for_profile(
        OFFICIAL_UNIFORM_PROFILE, play=True, no_root_pose=True
    ),
    rl_cfg=make_tracking_ppo_runner_cfg(experiment_name=NO_ROOT_POSE_EXPERIMENT_NAME),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id=HISTORY_CNN_TASK,
    env_cfg=make_history_cnn_tracking_env_cfg(),
    play_env_cfg=make_history_cnn_tracking_env_cfg(play=True),
    rl_cfg=make_history_cnn_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
