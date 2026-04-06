---
sidebar_position: 2
---

# Pico 4 VR Teleoperation

Real-time whole-body teleoperation using Pico 4 / Pico 4 Ultra full body tracking.

## Supported Devices

- Pico 4
- Pico 4 Ultra

## Architecture

```text
Pico Headset (XRoboToolkit App, Full Body Tracking)
    --WiFi--> PC (XRoboToolkit PC Service + xrobotoolkit_sdk)
                --> Teleopit (retarget -> RL policy -> MuJoCo / G1)
```

## Step 1: VR Headset Setup

1. Download the latest APK from [XRoboToolkit-Unity-Client Releases](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases)
2. Install via adb:
   ```bash
   adb install XRoboToolkit-Unity-Client.apk
   ```
3. Launch XRoboToolkit on the headset and enable **Full Body Tracking** mode
4. Ensure the headset and control PC are on the **same network**

## Step 2: PC Environment Setup

### Prerequisites

- Ubuntu 22.04
- Python 3.10+, Teleopit installed: `pip install -e .`
- pybind11: `conda install -c conda-forge pybind11`
- [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service):
  ```bash
  wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
  sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
  ```

### Install SDK

```bash
bash scripts/setup/setup_pico4.sh
```

The script will:
- Detect if `libPXREARobotSDK.so` is installed
- Register the dynamic library with the system linker
- Build and install `xrobotoolkit_sdk` Python bindings

Verify:
```bash
python -c "import xrobotoolkit_sdk; print('OK')"
```

## Step 3: Simulation Verification (Pico sim2sim)

Download model assets if you haven't already (see [Download Assets](../getting-started/download-assets)):

```bash
pip install modelscope
python scripts/setup/download_assets.py --only gmr ckpt bvh
```

Test VR tracking data and policy inference in MuJoCo first:

```bash
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=track.onnx
```

You should see the virtual robot following your VR movements. If the robot doesn't respond, check:
- XRoboToolkit shows "Connected" on the headset
- PC Service is running
- Both devices are on the same network

### Pause/Resume

- Press controller **A** to freeze tracking
- Press **A** again to clear the realtime reference buffer and smoothly resume live mocap

## Step 4: Hardware Deployment (Pico sim2real)

After verifying in simulation, connect to Unitree G1:

```bash
pip install -e '.[sim2real]'
git submodule update --init --recursive

python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    real_robot.network_interface=eth0
```

### Operation Flow

1. Start the script
2. Press **Start** on the remote -> enters `STANDING` (robot stands up)
3. Confirm Pico tracking data is arriving (check terminal logs)
4. Press **Y** -> enters `MOCAP` (teleoperation begins)
5. Press controller **A** to pause/resume tracking
6. Press **X** -> returns to `STANDING`
7. **L1+R1** -> emergency stop (`DAMPING`)

:::warning
When resuming from pause, keep still and match the paused pose as closely as possible. If distortion occurs, pause again immediately and adjust your pose before resuming.
:::

See [Sim2Real](sim2real) for the full state machine documentation.

## Common Parameters

```bash
# Adjust Pico wait timeout (default 60s)
input.pico4_timeout=30

# Adjust pause/resume transition
pause_resume_transition_duration=1.0

# Change policy frequency
policy_hz=30

# Change pause button
input.pause_button=right_axis_click

# Specify network interface
real_robot.network_interface=enp3s0
```

## Troubleshooting

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| `ImportError: xrobotoolkit_sdk` | SDK not installed | Run `bash scripts/setup/setup_pico4.sh` |
| `TimeoutError: No Pico4 body data` | Headset not connected or tracking not started | Check XRoboToolkit app status and network |
| Robot doesn't follow VR | Still in STANDING mode | Press **Y** on remote to enter MOCAP |
| `libPXREARobotSDK.so not found` | PC Service not installed | Install deb package and re-run setup script |
