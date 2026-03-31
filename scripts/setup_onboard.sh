#!/usr/bin/env bash
# One-click setup for onboard G1 sim2real environment.
#
# Installs:
#   1. CycloneDDS 0.10.x (native C library, built from source)
#   2. unitree_sdk2_python (via third_party submodule)
#   3. teleopit[onboard] Python package
#   4. Adds CYCLONEDDS_HOME to ~/.bashrc
#
# Prerequisites:
#   - Ubuntu 20.04 / 22.04
#   - conda environment activated (e.g. conda activate teleopit)
#   - Run from the Teleopit repo root:
#       bash scripts/setup_onboard.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CYCLONEDDS_INSTALL_DIR="$HOME/cyclonedds/install"

echo "=== Onboard G1 Sim2Real Setup ==="
echo "Repo root : $REPO_ROOT"
echo "CycloneDDS: $CYCLONEDDS_INSTALL_DIR"
echo ""

# ---------- 0. Check submodule ----------
if [ ! -f "$REPO_ROOT/third_party/unitree_sdk2_python/setup.py" ] && \
   [ ! -f "$REPO_ROOT/third_party/unitree_sdk2_python/pyproject.toml" ]; then
    echo "--- Initializing git submodules ---"
    git -C "$REPO_ROOT" submodule update --init --recursive
fi

# ---------- 1. System dependencies ----------
echo "--- Installing system build dependencies ---"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends cmake libssl-dev git build-essential

# ---------- 2. CycloneDDS native library ----------
if [ -f "$CYCLONEDDS_INSTALL_DIR/lib/libddsc.so" ]; then
    echo "--- CycloneDDS already installed at $CYCLONEDDS_INSTALL_DIR, skipping build ---"
else
    CYCLONEDDS_SRC="$HOME/cyclonedds"
    if [ ! -d "$CYCLONEDDS_SRC/.git" ]; then
        echo "--- Cloning CycloneDDS releases/0.10.x ---"
        git clone https://github.com/eclipse-cyclonedds/cyclonedds \
            -b releases/0.10.x --depth 1 "$CYCLONEDDS_SRC"
    else
        echo "--- CycloneDDS source already cloned ---"
    fi

    echo "--- Building CycloneDDS ---"
    mkdir -p "$CYCLONEDDS_SRC/build" "$CYCLONEDDS_INSTALL_DIR"
    cmake -S "$CYCLONEDDS_SRC" -B "$CYCLONEDDS_SRC/build" \
        -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_INSTALL_DIR"
    cmake --build "$CYCLONEDDS_SRC/build" --target install -- -j"$(nproc)"
    echo "--- CycloneDDS built and installed ---"
fi

# ---------- 3. Export CYCLONEDDS_HOME for current shell ----------
export CYCLONEDDS_HOME="$CYCLONEDDS_INSTALL_DIR"

# ---------- 4. Persist to ~/.bashrc ----------
BASHRC_LINE="export CYCLONEDDS_HOME=$CYCLONEDDS_INSTALL_DIR"
if ! grep -qF "CYCLONEDDS_HOME" "$HOME/.bashrc"; then
    echo "" >> "$HOME/.bashrc"
    echo "# CycloneDDS (for Unitree SDK2 / Teleopit onboard)" >> "$HOME/.bashrc"
    echo "$BASHRC_LINE" >> "$HOME/.bashrc"
    echo "--- Added CYCLONEDDS_HOME to ~/.bashrc ---"
else
    echo "--- CYCLONEDDS_HOME already in ~/.bashrc ---"
fi

# ---------- 5. Install unitree_sdk2_python ----------
echo "--- Installing unitree_sdk2_python ---"
uv pip install -e "$REPO_ROOT/third_party/unitree_sdk2_python"

# ---------- 6. Install teleopit[onboard] ----------
echo "--- Installing teleopit[onboard] ---"
uv pip install -e "$REPO_ROOT[onboard]"

# ---------- 7. Verify ----------
echo ""
echo "=== Verifying installation ==="
python -c "import cyclonedds; print('cyclonedds OK:', cyclonedds.__version__)"
python -c "import zmq; print('pyzmq OK:', zmq.__version__)"
python -c "import msgpack; print('msgpack OK:', msgpack.version)"
python -c "import unitree_sdk2py; print('unitree_sdk2py OK')"
python -c "from teleopit.inputs.zmq_provider import ZMQInputProvider; print('ZMQInputProvider OK')"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Download model assets:"
echo "       pip install modelscope"
echo "       python scripts/download_assets.py --only gmr ckpt bvh"
echo ""
echo "  2. Run onboard sim2real (replace IP with upper machine's IP):"
echo "       python scripts/run_onboard_sim2real.py \\"
echo "           controller.policy_path=track.onnx \\"
echo "           input.zmq_host=<上位机IP>"
