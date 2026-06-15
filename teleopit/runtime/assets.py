from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_ASSETS_ROOT = PROJECT_ROOT / "assets" / "robots"
GMR_ASSETS_ROOT = PROJECT_ROOT / "teleopit" / "retargeting" / "gmr" / "assets"
UNITREE_G1_XML = ROBOT_ASSETS_ROOT / "unitree_g1" / "g1_29dof.xml"
UNITREE_G1_MJLAB_XML = UNITREE_G1_XML


def missing_gmr_assets_message(path: str | Path, *, label: str = "Required asset") -> str:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return (
        f"{label} not found: {resolved}\n"
        "Download the external robot assets with:\n"
        "  python scripts/setup/download_assets.py --only robots gmr"
    )
