#!/usr/bin/env bash
# One-click setup for onboard G1 sim2real environment.
#
# Installs:
#   1. System build dependencies (cmake, etc.)
#   2. g1_bridge_sdk (C++ DDS bridge via pybind11)
#   3. teleopit[onboard] Python package
#
# Prerequisites:
#   - Ubuntu 20.04 / 22.04
#   - conda environment activated (e.g. conda activate teleopit)
#   - Run from the Teleopit repo root:
#       bash scripts/setup/setup_onboard.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Onboard G1 Sim2Real Setup ==="
echo "Repo root : $REPO_ROOT"
echo ""

# ---------- 1. System dependencies ----------
echo "--- Installing system build dependencies ---"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends cmake libssl-dev git build-essential

# ---------- 2. Install g1_bridge_sdk (C++ DDS bridge) ----------
echo "--- Installing pybind11 ---"
uv pip install pybind11

echo "--- Building and installing g1_bridge_sdk ---"
uv pip install "$REPO_ROOT/third_party/g1_bridge_sdk"

# ---------- 3. Install teleopit[onboard] ----------
echo "--- Installing teleopit[onboard] ---"
uv pip install -e "$REPO_ROOT[onboard]"

# ---------- 4. Verify ----------
echo ""
echo "=== Verifying installation ==="
python -c "import g1_bridge_sdk; print('g1_bridge_sdk OK')"
python -c "import zmq; print('pyzmq OK:', zmq.__version__)"
python -c "import msgpack; print('msgpack OK:', msgpack.version)"
python -c "from teleopit.inputs.zmq_provider import ZMQInputProvider; print('ZMQInputProvider OK')"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Download model assets:"
echo "       pip install modelscope"
echo "       python scripts/setup/download_assets.py --only gmr ckpt bvh"
echo ""
echo "  2. Run onboard sim2real (replace IP with upper machine's IP):"
echo "       python scripts/run/run_onboard_sim2real.py \\"
echo "           controller.policy_path=track.onnx \\"
echo "           input.zmq_host=<上位机IP>"
