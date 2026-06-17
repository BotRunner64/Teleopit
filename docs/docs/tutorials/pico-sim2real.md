---
sidebar_position: 4
---

# Pico 4 VR Teleoperation on Unitree G1

Use this tutorial after [Pico Sim2Sim](pico-sim2sim) is working. It deploys the
same realtime Pico input path to a physical Unitree G1.

```text
Pico headset -> Teleopit host -> retarget -> RL policy -> g1_bridge_sdk -> G1
```

There are two deployment styles:

| Deployment | Where Teleopit Runs | Main Difference |
|------------|---------------------|-----------------|
| Wired PC-to-G1 | External workstation or laptop | Set `real_robot.network_interface` to the PC Ethernet interface connected to G1 |
| Onboard | G1 onboard computer | Install Teleopit on the onboard computer; `eth0` is usually correct |

Both styles use `Pico4InputProvider` and the in-process pico-bridge receiver.
There is no separate onboard Pico input mode.

Teleopit targets pico-bridge 0.2.1 and its `pico_native` tracking semantics.

## 1. Install Runtime Dependencies

Install Pico and sim2real dependencies on the machine that will run Teleopit:

```bash
pip install -e '.[pico4]'
git submodule update --init --recursive
bash scripts/setup/setup_g1_bridge.sh
```

Verify Pico receiver import:

```bash
python -c "from pico_bridge import PicoBridge; print('OK')"
```

## 2. Choose The Network Interface

`real_robot.network_interface` is the Linux interface used for Unitree DDS
communication.

For wired PC-to-G1 deployment:

1. Connect the PC to the G1 by Ethernet.
2. Run `ifconfig` on the PC.
3. Use the Ethernet interface connected to the robot, for example `enp130s0`.
4. Keep the Pico headset on a network that can reach the PC running Teleopit.

For onboard deployment:

1. Run Teleopit on the robot onboard computer.
2. Keep the Pico headset on a network that can reach the onboard computer.
3. Use `real_robot.network_interface=eth0` unless your robot network differs.
4. Set `input.bridge_advertise_ip=<host-ip>` if Pico discovery advertises the
   wrong address.

### Onboard RealSense On Arm

The pico-bridge PC receiver supports Arm machines when the required Python
dependencies are available. On Arm onboard computers that need RealSense preview,
install `pyrealsense2` from conda-forge in the active Conda environment instead
of relying on the pip package:

```bash
pip uninstall pyrealsense2
conda install -c conda-forge pyrealsense2
```

This only matters when using the optional RealSense preview path
(`input.video.enabled=true`). Pico tracking and robot control do not require
RealSense.

## 3. Run The Controller

Wired PC example:

```bash
python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    real_robot.network_interface=enp130s0
```

Onboard example:

```bash
python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    real_robot.network_interface=eth0
```

## Operator Flow

Keep the Unitree remote in hand. `L1+R1` is the emergency stop path into
`DAMPING`.

| Control | Action |
|---------|--------|
| Unitree remote `Start` | Enter `STANDING` |
| Unitree remote `Y` | Enter `MOCAP` |
| Pico/controller `A` | Pause / resume live mocap |
| Pico/controller `B` | Toggle `MOCAP` / `ARMS` |
| Unitree remote `X` | Return to `STANDING` |
| Unitree remote `L1+R1` | Emergency stop (`DAMPING`) |

Enter `MOCAP` only after Pico tracking is stable. Teleopit validates consecutive
mocap frames before switching; if validation fails, the robot stays in
`STANDING`.

## Runtime Behavior

Pico sim2real uses the shared realtime reference timeline:

```text
Pico body frames -> retarget -> reference buffer -> observation -> policy -> G1 joints
```

When entering `STANDING`, Teleopit releases active Unitree modes, enters
debug/low-level control, locks the current joints briefly, resets policy state,
and ramps Kp without changing policy targets.

When entering `MOCAP`, Teleopit resets policy/reference state and starts tracking
the live mocap command through the realtime reference timeline.

`ARMS` keeps the same live retargeting timeline running, but sends the motion
tracker a composed reference: body, waist, and legs stay at the standing pose
while both arms follow the live retargeted result. Entering or leaving `ARMS`
resets policy/reference alignment and uses the same Kp ramp safety path.

## Pause / Resume

Pico pause/resume is a mocap-session control event.

- `ACTIVE`: the pause button freezes the current reference pose.
- `PAUSED`: pressing it again clears policy/reference state, warms the realtime
  buffer, re-centers yaw/XY alignment, and resumes from live mocap.

:::warning
Resume while standing still and close to the paused pose. This reduces sudden
reference changes when live tracking resumes.
:::

## Optional LinkerHand Control

Pico sim2real can drive LinkerHand hands from Pico input:

- `gripper`: hold the matching side grip as a deadman switch; the matching
  trigger closes that hand. This mode supports `hands.driver=linkerhand_l6` and
  `hands.driver=linkerhand_o6`; speed and open/close poses come from the matching
  driver config.
- `vr_hand_pose`: L6-only mode that retargets Pico hand pose through somehand and
  commands the continuous L6 hand target. If a hand pose disappears, that side
  keeps its last commanded pose. This mode uses Teleopit's Pico landmark adapter
  and the public `somehand.api` from somehand 0.2.0. It always sets L6 speed to
  the maximum.

Hand control is active in `MOCAP` and `ARMS`. It sends the open pose in
`STANDING`, `DAMPING`, paused mocap, and shutdown.

Install the dexhand extra first if it was not installed with the main Pico
profile:

```bash
pip install -e '.[dexhand]'
scripts/setup/download_somehand_l6_assets.sh
```

Bring up the CAN interfaces before testing or running hand control:

```bash
sudo /usr/sbin/ip link set can0 up type can bitrate 1000000
sudo /usr/sbin/ip link set can1 up type can bitrate 1000000
```

Before enabling full sim2real, verify the hand connection with a standalone
open/close test. The test runs until Ctrl-C:

```bash
python scripts/dev/test_linkerhand_l6.py \
    --hand-type both \
    --left-can can0 \
    --right-can can1
```

For an O6 standalone open/close test, add the O6 driver:

```bash
python scripts/dev/test_linkerhand_l6.py \
    --driver linkerhand_o6 \
    --hand-type both \
    --left-can can0 \
    --right-can can1
```

To test O6 with live Pico gripper input, add `--mode gripper`.

Then enable L6 gripper control in Pico sim2real:

```bash
hands.enabled=true
hands.driver=linkerhand_l6
hands.mode=gripper
hands.linkerhand_l6.left_can=can0
hands.linkerhand_l6.right_can=can1
```

For O6 gripper control, use:

```bash
hands.enabled=true
hands.driver=linkerhand_o6
hands.mode=gripper
hands.linkerhand_o6.left_can=can0
hands.linkerhand_o6.right_can=can1
```

For continuous VR hand-pose control, use:

```bash
hands.enabled=true
hands.driver=linkerhand_l6
hands.mode=vr_hand_pose
hands.linkerhand_l6.left_can=can0
hands.linkerhand_l6.right_can=can1
```

## Optional RealSense Preview

Stream the G1 RealSense color camera back to the Pico headset:

```bash
python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=track.onnx \
    real_robot.network_interface=enp130s0 \
    input.video.enabled=true \
    input.video.device=<optional-realsense-serial>
```

If video fails, control continues unless `input.video.fail_on_error=true`.

## Common Parameters

```bash
# Real G1 DDS interface
real_robot.network_interface=enp130s0

# Pico timeout
input.pico4_timeout=30

# Override advertised Pico discovery IP
input.bridge_advertise_ip=192.168.1.20

# Consecutive valid mocap frames required before MOCAP
mocap_switch.check_frames=10

# Change Pico pause button
input.pause_button=right_axis_click

# Enable LinkerHand gripper control
hands.enabled=true
hands.driver=linkerhand_l6
hands.mode=gripper

# Enable headset video preview
input.video.enabled=true
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No LowState received | Wrong interface or G1 network not connected | Check Ethernet wiring and `real_robot.network_interface` |
| `TimeoutError: No Pico4 body data` | Headset is not connected or tracking is inactive | Check headset app, network, and `input.pico4_timeout` |
| Cannot enter debug mode | Unitree mode release failed | Stop other robot modes and press `Start` again |
| Robot enters `STANDING` but not `MOCAP` | Mocap validation failed | Keep tracking active and stable; check `mocap_switch.check_frames` logs |
| Pico pause does not return to `STANDING` | Expected behavior | Pico pause freezes mocap; press remote `X` for `STANDING` |
| LinkerHand does not move | `hands.enabled=false`, not in `MOCAP`, gripper deadman released, SDK/assets not installed, or CAN channel wrong | Enable `hands.enabled`, enter `MOCAP`, run `scripts/dev/test_linkerhand_l6.py`, and check the selected driver's `left_can` / `right_can` |
| Video preview is unavailable | RealSense or video source failed | Check camera permissions, `input.video.source`, and logs |
