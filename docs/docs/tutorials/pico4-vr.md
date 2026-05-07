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
Pico Headset (pico-bridge client, full body tracking)
    --WiFi--> PC (pico-bridge PC receiver)
                --> Teleopit (retarget -> RL policy -> MuJoCo / G1)
```

## Step 1: VR Headset Setup

1. Download the headset client APK from [pico-bridge Releases](https://github.com/BotRunner64/pico-bridge/releases).
2. Install via adb:
   ```bash
   adb install pico-bridge.apk
   ```
3. Launch the pico-bridge headset client and enable full body tracking.
4. Ensure the headset and control PC are on the **same network**

## Step 2: PC Environment Setup

### Prerequisites

- Ubuntu 22.04
- Python 3.10+
- Teleopit with the Pico extra: `pip install -e '.[pico4]'`

### Install pico-bridge PC Receiver

```bash
pip install -e '.[pico4]'
```

Verify:
```bash
python -c "from pico_bridge import PicoBridge; print('OK')"
```

Teleopit starts the PC receiver from `Pico4InputProvider`.

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
- The pico-bridge headset client shows a connection to the PC receiver
- `input.bridge_host`, `input.bridge_port`, and optional `input.bridge_advertise_ip` match your network
- Both devices are on the same network

### Keyboard Mode Flow

With `teleopit/configs/pico4_sim.yaml`, realtime keyboard mode is enabled by default:

- Press **Y** to enter `MOCAP`
- Press **A** to pause/resume live mocap
- Press **X** to return to `STANDING`
- Press **Q** to quit the simulation loop

The loop starts directly in `STANDING`, so you can press **Y** once tracking is ready.

### Pause/Resume

- Press keyboard **A** or Pico controller **A** to freeze tracking
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

# Disable realtime keyboard mode state machine
keyboard.enabled=false

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
| `ImportError: pico_bridge` | PC receiver package not installed | Run `pip install -e '.[pico4]'` |
| `TimeoutError: No Pico4 body data` | Headset not connected or tracking not started | Check pico-bridge headset app status and network |
| Robot doesn't follow VR | Still in STANDING mode | Press **Y** on remote to enter MOCAP |
| Discovery does not find the PC | Wrong network interface or blocked UDP | Set `input.bridge_advertise_ip=<pc-ip>` and confirm UDP port `63901` is reachable |
