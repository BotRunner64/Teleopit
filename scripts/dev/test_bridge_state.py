#!/usr/bin/env python3
"""Diagnostic: compare C++ bridge state vs Python SDK state."""
import sys
import time
import copy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "unitree_sdk2_python"))

import numpy as np

NETWORK = sys.argv[1] if len(sys.argv) > 1 else "enp130s0"
NUM_JOINTS = 29

# ---- Method 1: C++ bridge ----
import g1_bridge_sdk

bridge = g1_bridge_sdk.G1Bridge(NETWORK)
if not bridge.wait_for_state(5.0):
    print("ERROR: C++ bridge no state")
    sys.exit(1)

time.sleep(0.5)  # let a few callbacks arrive

qpos_cpp, qvel_cpp, quat_cpp, angvel_cpp = bridge.get_state()
remote_cpp = bridge.get_wireless_remote()

print("=== C++ bridge ===")
print(f"qpos[:6]  = {qpos_cpp[:6]}")
print(f"qvel[:6]  = {qvel_cpp[:6]}")
print(f"quat      = {quat_cpp}")
print(f"ang_vel   = {angvel_cpp}")
print(f"remote[:8]= {list(remote_cpp[:8])}")
print(f"qpos_norm = {np.linalg.norm(qpos_cpp):.4f}")
print()

# ---- Method 2: Python SDK (on same DDS domain) ----
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HG_LowState

ChannelFactoryInitialize(0, NETWORK)

lowstate = [None]

def on_lowstate(msg):
    lowstate[0] = copy.deepcopy(msg)

sub = ChannelSubscriber("rt/lowstate", HG_LowState)
sub.Init(on_lowstate, 10)

deadline = time.monotonic() + 5.0
while lowstate[0] is None and time.monotonic() < deadline:
    time.sleep(0.05)

if lowstate[0] is None:
    print("ERROR: Python SDK no state")
    sys.exit(1)

time.sleep(0.5)
ls = lowstate[0]

qpos_py = np.array([ls.motor_state[i].q for i in range(NUM_JOINTS)], dtype=np.float32)
qvel_py = np.array([ls.motor_state[i].dq for i in range(NUM_JOINTS)], dtype=np.float32)
quat_py = np.array(ls.imu_state.quaternion, dtype=np.float32)
angvel_py = np.array(ls.imu_state.gyroscope, dtype=np.float32)

print("=== Python SDK ===")
print(f"qpos[:6]  = {qpos_py[:6]}")
print(f"qvel[:6]  = {qvel_py[:6]}")
print(f"quat      = {quat_py}")
print(f"ang_vel   = {angvel_py}")
print(f"qpos_norm = {np.linalg.norm(qpos_py):.4f}")
print()

# ---- Compare ----
print("=== Diff ===")
print(f"qpos max_diff = {np.max(np.abs(qpos_cpp - qpos_py)):.6f}")
print(f"qvel max_diff = {np.max(np.abs(qvel_cpp - qvel_py)):.6f}")
print(f"quat max_diff = {np.max(np.abs(quat_cpp - quat_py)):.6f}")

if np.linalg.norm(qpos_cpp) < 0.01 and np.linalg.norm(qpos_py) > 0.1:
    print("\n*** C++ qpos is near-zero but Python is not — bridge state reading is BROKEN ***")
elif np.max(np.abs(qpos_cpp - qpos_py)) > 0.01:
    print("\n*** Significant qpos mismatch between C++ and Python ***")
else:
    print("\n*** State data matches — issue is elsewhere ***")
