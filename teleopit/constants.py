"""Robot-specific dimension constants for the Unitree G1 pipeline."""

ROOT_POS_DIM = 3
ROOT_QUAT_DIM = 4
ROOT_DIM = ROOT_POS_DIM + ROOT_QUAT_DIM  # 7: pos(3) + quat_wxyz(4)
NUM_JOINTS = 29  # G1 actuated joints
FULL_QPOS_DIM = ROOT_DIM + NUM_JOINTS  # 36: root + joints
