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
    --WiFi--> Teleopit host (pico-bridge receiver)
                --> Teleopit (retarget -> RL policy -> MuJoCo / G1)
```

The Teleopit host can be a workstation PC or the robot onboard computer. Teleopit starts
the `pico_bridge.PicoBridge` receiver in-process, so onboard deployment uses the same
Pico input mode as PC deployment.

## Step 1: VR Headset Setup

1. Download the headset client APK from [pico-bridge Releases](https://github.com/BotRunner64/pico-bridge/releases).
2. Install via adb:
   ```bash
   adb install pico-bridge.apk
   ```
3. Launch the pico-bridge headset client and enable full body tracking.
4. Ensure the headset and Teleopit host are on the **same network**

## Step 2: Teleopit Host Environment Setup

### Prerequisites

- Ubuntu 22.04
- Python 3.10+
- Teleopit with the Pico extra: `pip install -e '.[pico4]'`

### Install pico-bridge Receiver

```bash
pip install -e '.[pico4]'
```

Verify:
```bash
python -c "from pico_bridge import PicoBridge; print('OK')"
```

Teleopit starts the receiver from `Pico4InputProvider`. The Pico extra installs
pico-bridge 0.2.0 with camera support.

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
- The pico-bridge headset client shows a connection to the Teleopit host receiver
- `input.bridge_host`, `input.bridge_port`, and optional `input.bridge_advertise_ip` match your network
- Both devices are on the same network

### Optional Pico Video Preview

pico-bridge 0.2.0 can send a host camera preview back to the headset. Teleopit keeps this
disabled by default so tracking and control still run on hosts without camera access.

For sim2sim, stream the MuJoCo `d435i_rgb` camera:

```bash
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=track.onnx \
    input.video.enabled=true
```

For sim2real, stream the G1 RealSense color camera:

```bash
python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    input.video.enabled=true \
    input.video.device=<optional-realsense-serial>
```

If the video source fails, Teleopit logs the error, disables video, and keeps mocap/control running.
Set `input.video.fail_on_error=true` to make video startup fail-fast.

### Keyboard Mode Flow

With `teleopit/configs/pico4_sim.yaml`, realtime keyboard mode is enabled by default:

- Press **Y** to enter `MOCAP`
- Press **A** to pause/resume live mocap
- Press **X** to return to `STANDING`
- Press **Q** to quit the simulation loop

The loop starts directly in `STANDING`, so you can press **Y** once tracking is ready.

### Pause/Resume

- Press keyboard **A** or Pico controller **A** to freeze tracking
- Press **A** again to resume tracking. Teleopit re-centers heading and ground-plane position before following again.

## Step 4: Hardware Deployment (Pico sim2real)

After verifying in simulation, connect to Unitree G1:

```bash
pip install -e '.[sim2real]'
git submodule update --init --recursive

python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    real_robot.network_interface=enp130s0
```

For wired control from a PC, run `ifconfig` on the PC and set `real_robot.network_interface` to the Ethernet interface connected to the G1, such as `enp130s0`. For onboard execution on the robot computer, the default `eth0` is usually correct.

### Operation Flow

1. Start the script
2. Press **Start** on the remote -> enters `STANDING` (robot stands up)
3. Confirm Pico tracking data is arriving (check terminal logs)
4. Press **Y** -> enters `MOCAP` (teleoperation begins)
5. Press controller **A** to pause/resume tracking
6. Press **X** -> returns to `STANDING`
7. **L1+R1** -> emergency stop (`DAMPING`)

:::warning
When resuming from pause, keep still and stay as close as practical to the paused pose while new mocap frames arrive. This reduces sudden reference changes when tracking resumes.
:::

See [Sim2Real](sim2real) for the full state machine documentation.

## Common Parameters

```bash
# Adjust Pico wait timeout (default 60s)
input.pico4_timeout=30

# Adjust how many fresh tracking frames are collected before resume
pause_resume_warmup_steps=2

# Disable realtime keyboard mode state machine
keyboard.enabled=false

# Change policy frequency
policy_hz=30

# Change pause button
input.pause_button=right_axis_click

# Enable host camera preview in the Pico headset
input.video.enabled=true

# Specify network interface
real_robot.network_interface=enp130s0
```

## Troubleshooting

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| `ImportError: pico_bridge` | Receiver package not installed | Run `pip install -e '.[pico4]'` |
| `TimeoutError: No Pico4 body data` | Headset not connected or tracking not started | Check pico-bridge headset app status and network |
| Robot doesn't follow VR | Still in STANDING mode | Press **Y** on remote to enter MOCAP |
| Discovery does not find the host | Wrong network interface or blocked UDP | Set `input.bridge_advertise_ip=<host-ip>` and confirm UDP port `63901` is reachable |
| Pico video preview is black or unavailable | Camera source failed or video disabled | Set `input.video.enabled=true`, check RealSense access, and review logs |
