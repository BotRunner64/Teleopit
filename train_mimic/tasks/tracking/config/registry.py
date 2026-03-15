"""Registry wiring for the official tracking task surface."""

from mjlab.tasks.registry import register_mjlab_task

from train_mimic.tasks.tracking.config.constants import (
    LEGACY_TASK_ALIAS,
    OFFICIAL_TASK,
    VELOCITY_EXPERIMENT_NAME,
    VELOCITY_TRACKING_TASK,
)
from train_mimic.tasks.tracking.config.env import (
    make_tracking_env_cfg_for_profile,
    make_vel_tracking_env_cfg_for_profile,
)
from train_mimic.tasks.tracking.config.profiles import (
    OFFICIAL_UNIFORM_PROFILE,
    VELOCITY_DRIVEN_PROFILE,
)
from train_mimic.tasks.tracking.config.rl import make_tracking_ppo_runner_cfg
from train_mimic.tasks.tracking.rl import MotionTrackingOnPolicyRunner


def _register(task_id: str) -> None:
    register_mjlab_task(
        task_id=task_id,
        env_cfg=make_tracking_env_cfg_for_profile(OFFICIAL_UNIFORM_PROFILE),
        play_env_cfg=make_tracking_env_cfg_for_profile(OFFICIAL_UNIFORM_PROFILE, play=True),
        rl_cfg=make_tracking_ppo_runner_cfg(),
        runner_cls=MotionTrackingOnPolicyRunner,
    )


_register(OFFICIAL_TASK)
_register(LEGACY_TASK_ALIAS)

register_mjlab_task(
    task_id=VELOCITY_TRACKING_TASK,
    env_cfg=make_vel_tracking_env_cfg_for_profile(VELOCITY_DRIVEN_PROFILE),
    play_env_cfg=make_vel_tracking_env_cfg_for_profile(VELOCITY_DRIVEN_PROFILE, play=True),
    rl_cfg=make_tracking_ppo_runner_cfg(experiment_name=VELOCITY_EXPERIMENT_NAME),
    runner_cls=MotionTrackingOnPolicyRunner,
)
