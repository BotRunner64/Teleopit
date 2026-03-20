"""Public constants for supported tracking tasks."""

DEFAULT_TRAIN_MOTION_FILE = "data/datasets/twist2_full/train.npz"
VELCMD_HISTORY_TASK = "Tracking-Flat-G1-VelCmdHistory"
VELCMD_HISTORY_EXPERIMENT_NAME = "g1_tracking_velcmd_history"
MOTION_TRACKING_DEPLOY_TASK = "Tracking-Flat-G1-MotionTrackingDeploy"
MOTION_TRACKING_DEPLOY_EXPERIMENT_NAME = "g1_tracking_motion_tracking_deploy"

SUPPORTED_TASKS = (
    VELCMD_HISTORY_TASK,
    MOTION_TRACKING_DEPLOY_TASK,
)
