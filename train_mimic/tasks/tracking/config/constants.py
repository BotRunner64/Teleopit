"""Public constants for supported tracking tasks."""

DEFAULT_TRAIN_MOTION_FILE = "data/datasets/twist2_full/train.npz"
VELCMD_HISTORY_TASK = "Tracking-Flat-G1-VelCmdHistory"
VELCMD_HISTORY_EXPERIMENT_NAME = "g1_tracking_velcmd_history"
VELCMD_HISTORY_ADAPTIVE_TASK = "Tracking-Flat-G1-VelCmdHistoryAdaptive"
VELCMD_HISTORY_ADAPTIVE_EXPERIMENT_NAME = "g1_tracking_velcmd_history_adaptive"
VELCMD_REF_WINDOW_TASK = "Tracking-Flat-G1-VelCmdRefWindow"
VELCMD_REF_WINDOW_EXPERIMENT_NAME = "g1_tracking_velcmd_ref_window"
MOTION_TRACKING_DEPLOY_TASK = "Tracking-Flat-G1-MotionTrackingDeploy"
MOTION_TRACKING_DEPLOY_EXPERIMENT_NAME = "g1_tracking_motion_tracking_deploy"
VELCMD_HISTORY_DEPLOY_TASK = "Tracking-Flat-G1-VelCmdHistoryDeploy"
VELCMD_HISTORY_DEPLOY_EXPERIMENT_NAME = "g1_tracking_velcmd_history_deploy"

SUPPORTED_TASKS = (
    VELCMD_HISTORY_TASK,
    VELCMD_HISTORY_ADAPTIVE_TASK,
    VELCMD_REF_WINDOW_TASK,
    MOTION_TRACKING_DEPLOY_TASK,
    VELCMD_HISTORY_DEPLOY_TASK,
)
