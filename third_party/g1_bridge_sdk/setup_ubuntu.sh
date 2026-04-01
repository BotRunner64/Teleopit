#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Get unitree_sdk2 C++ SDK
if [ ! -d "thirdparty/unitree_sdk2" ]; then
    echo "Cloning unitree_sdk2 ..."
    mkdir -p thirdparty
    git clone https://github.com/unitreerobotics/unitree_sdk2.git thirdparty/unitree_sdk2
fi

# 2. Install pybind11
pip install pybind11

# 3. Build and install
pip uninstall -y g1_bridge_sdk 2>/dev/null || true
pip install .

echo "Done! 'import g1_bridge_sdk' should work now."
