---
sidebar_position: 4
---

# G1 Bridge SDK

C++ DDS bridge library wrapping unitree_sdk2 with pybind11, providing near-zero latency (< 0.5 ms) access to Unitree G1's real-time communication interface.

All DDS publish/subscribe runs on native C++ threads. The Python side only calls simple get/set methods.

## Dependencies

- CMake >= 3.10
- GCC >= 9.4 (C++17 support)
- pybind11 >= 2.6
- Unitree SDK2 (bundled in `third_party/g1_bridge_sdk/thirdparty/unitree_sdk2/`, no manual install needed)
- Cyclone DDS (unitree_sdk2 dependency)

## Installation

```bash
pip install pybind11
pip install third_party/g1_bridge_sdk
```

CMake build triggers automatically during install. If compilation fails:

```bash
bash third_party/g1_bridge_sdk/setup_ubuntu.sh
```

## Python API

```python
import g1_bridge_sdk

bridge = g1_bridge_sdk.G1Bridge(
    network_interface="eth0",   # Network interface connected to G1
    publish_hz=200              # Command publish rate (default 200 Hz)
)
```

| Method | Description |
|--------|-------------|
| `wait_for_state(timeout_sec=5.0)` | Block until first LowState frame; returns False on timeout |
| `get_state()` | Returns `(qpos[29], qvel[29], quat[4], ang_vel[3])` numpy arrays |
| `get_state_counter()` | Returns cumulative LowState frame count |
| `get_wireless_remote()` | Returns 40-byte wireless remote data |
| `get_mode_machine()` | Returns current mode_machine value |
| `set_target(target, kp, kd)` | Set target joint positions and PD gains (29 elements each) |
| `lock_joints()` | Lock current joint positions |
| `set_damping()` | Switch to damping mode (for emergency stop) |
| `start_publish()` | Start command publish thread |
| `stop_publish()` | Stop command publish thread |
| `check_mode()` | Query current motion mode, returns `(code, name)` |
| `select_mode(name)` | Switch motion mode (e.g., `"ai"`, `"normal"`) |
| `release_mode()` | Release current mode, enter low-level control |

## Usage

- **Onboard sim2real**: `scripts/run/run_onboard_sim2real.py`
- **Pico4 hardware teleoperation**: `scripts/run/run_sim2real.py`
- **Standalone standing test**: `scripts/run/standalone_standing.py`
