---
sidebar_position: 4
---

# Onboard Sim2Real

Run the sim2real control loop directly on the G1's onboard NX computer, with Pico 4 tracking data relayed via ZMQ from an external PC.

## Network Topology

A router connecting both PCs via WLAN/WiFi:

```text
 [Router] --- WLAN/WiFi --- [PC2: User PC]
    |
    +--- WLAN/WiFi --- [PC1: G1 NX Onboard]
```

- **PC1** (G1 NX): Wirelessly connected to the router
- **PC2** (User PC): Wirelessly connected to the same router

## Setup Fixed IPs

1. Assign fixed IPs in the router admin page, e.g.:
   - PC1 (NX): `192.168.1.101`
   - PC2 (User PC): `192.168.1.102`
2. Or manually configure static IPs on each device

## Verify Connectivity

```bash
# From PC2, ping PC1
ping 192.168.1.101

# SSH from PC2 to PC1
ssh user@192.168.1.101
```

## Install Environment (on NX)

```bash
# SSH into NX, then run in the Teleopit repo directory
bash scripts/setup/setup_onboard.sh
```

This script installs system dependencies, builds `g1_bridge_sdk`, and installs `teleopit[onboard]`.

## Run Onboard Sim2Real

**On PC2 (User PC)** — start the Pico 4 ZMQ publisher:

```bash
python scripts/dev/zmq_pico4_publisher.py --bind 0.0.0.0 --port 5555
```

This reads Pico 4 body tracking data and publishes it over ZMQ. Keep it running.

**On PC1 (NX)** — start the onboard control loop:

```bash
python scripts/run/run_onboard_sim2real.py \
    controller.policy_path=track.onnx \
    real_robot.network_interface=wlan0 \
    input.zmq_host=192.168.1.102
```

Where `input.zmq_host` is PC2's IP address (the machine running the ZMQ publisher).

## Operation

The control flow and remote controller mapping are the same as [Sim2Real Deployment](sim2real#remote-controller-mapping). The only difference is that the control loop runs on the NX instead of an external PC.
