---
sidebar_position: 2
---

# Pico 4 VR Teleoperation in Simulation

Use this tutorial to verify Pico 4 / Pico 4 Ultra full-body tracking in MuJoCo
before running on real Unitree G1 hardware.

```text
Pico headset -> pico-bridge receiver -> retarget -> RL policy -> MuJoCo G1
```

After this works, continue with [Pico Sim2Real](pico-sim2real).

## Supported Devices

- Pico 4
- Pico 4 Ultra

## 1. Set Up The Headset

1. Download the headset APK from [pico-bridge Releases](https://github.com/BotRunner64/pico-bridge/releases).
2. Install it with adb:
   ```bash
   adb install pico-bridge.apk
   ```
3. Launch the pico-bridge headset client.
4. Enable full-body tracking.
5. Keep the headset and Teleopit host on the same network.

## 2. Install The Pico Host Extra

On the machine that will run Teleopit:

```bash
pip install -e '.[pico4]'
```

Verify the receiver package:

```bash
python -c "from pico_bridge import PicoBridge; print('OK')"
```

Teleopit starts `pico_bridge.PicoBridge` in-process through
`Pico4InputProvider`. The same Pico input path is used later for wired and
onboard sim2real deployment.

For Teleopit 0.3.0, keep the host receiver on pico-bridge 0.2.0. pico-bridge
0.2.1 changes interface semantics and is not the supported receiver version for
this Teleopit release.

## 3. Download Assets

```bash
pip install modelscope
python scripts/setup/download_assets.py --only gmr ckpt bvh
```

## 4. Run Pico Sim2Sim

```bash
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=track.onnx
```

The simulation starts in `STANDING`. Wait until Pico tracking is active, then
enter `MOCAP`.

| Keyboard | Action |
|----------|--------|
| `Y` | Enter `MOCAP` |
| `A` | Pause / resume live mocap |
| `X` | Return to `STANDING` |
| `Q` | Quit |

`pico4_sim.yaml` defaults to `viewers=all`, which opens mocap, retarget, and
sim2sim viewers. Use `viewers=sim2sim` or `viewers=none` when you want fewer
windows.

## Pause / Resume

Pico pause/resume freezes the mocap session; it is not a switch back to
`STANDING`.

- Press keyboard `A` or the Pico/controller pause button to freeze the current
  reference pose.
- Press it again to rebuild the realtime reference path, re-center yaw and
  ground-plane position, and continue from the current live tracking stream.

The default Pico pause button is `A`. Supported overrides include `B`, `X`, `Y`,
`left_axis_click`, `right_axis_click`, `left_menu_button`, and
`right_menu_button`.

## Optional Headset Video Preview

pico-bridge 0.2.0 can show a host-side camera stream in the headset. In
simulation, Teleopit can stream the MuJoCo `d435i_rgb` camera:

```bash
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=track.onnx \
    input.video.enabled=true
```

Use `input.video.source=test-pattern` for a receiver-side video sanity check. If
video startup fails, Teleopit logs the error, disables video, and keeps tracking
and control running. Set `input.video.fail_on_error=true` to fail startup
instead.

## Common Parameters

```bash
# Pico wait timeout for the first body frame
input.pico4_timeout=30

# Override the IP advertised to the headset during discovery
input.bridge_advertise_ip=192.168.1.20

# Disable discovery and bind explicitly
input.bridge_discovery=false input.bridge_host=0.0.0.0 input.bridge_port=63901

# Change the Pico pause button
input.pause_button=right_axis_click

# Disable keyboard mode control
keyboard.enabled=false

# Change policy frequency
policy_hz=30

# Enable headset video preview
input.video.enabled=true
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ImportError: pico_bridge` | Pico extra not installed | Run `pip install -e '.[pico4]'` |
| Startup says pico-bridge is too old | Installed receiver does not support video args | Reinstall the Pico extra so pico-bridge 0.2.0 is used |
| `TimeoutError: No Pico4 body data` | Headset is not connected or body tracking is inactive | Check the headset app, network, and `input.pico4_timeout` |
| Discovery cannot find the host | Wrong advertised IP or blocked UDP | Set `input.bridge_advertise_ip=<host-ip>` and confirm UDP port `63901` is reachable |
| Sim robot does not follow | Loop is still in `STANDING` | Press `Y` after tracking is ready |
| Pico video is black or disabled | Video source failed or camera access is unavailable | Check `input.video.source` and logs |
