#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOMEHAND_DIR="${PROJECT_ROOT}/third_party/somehand"

if [[ ! -f "${SOMEHAND_DIR}/scripts/setup/download_assets.py" ]]; then
  echo "somehand submodule is not initialized. Run: git submodule update --init third_party/somehand" >&2
  exit 1
fi

cd "${SOMEHAND_DIR}"
python scripts/setup/download_assets.py --only mjcf "$@"
