"""Tests for hc_mocap BVH format parsing and loading.

Covers:
- read_bvh() parsing hc_mocap format (joint count, frame count, quaternion shape)
- _load_bvh_file() with hc_mocap (scale, LeftFootMod, fps)
- BVHInputProvider with hc_mocap (get_frame() returns valid HumanFrame)
"""

import numpy as np
import pytest

from teleopit.retargeting.gmr.utils.lafan_vendor.extract import read_bvh
from teleopit.inputs.bvh_provider import BVHInputProvider


class TestHCMocapParsing:
    """Test read_bvh() with hc_mocap format."""

    def test_read_bvh_hc_mocap_parses(self):
        """Verify hc_mocap BVH parses correctly: 50 joints, 934 frames, quaternion shape."""
        bvh_path = "data/motion_corrected_v2.bvh"
        data = read_bvh(bvh_path)

        # Check joint count
        assert len(data.bones) == 50, f"Expected 50 joints, got {len(data.bones)}"

        # Check frame count
        assert data.quats.shape[0] == 934, f"Expected 934 frames, got {data.quats.shape[0]}"

        # Check quaternion shape (frames, joints, 4)
        assert data.quats.shape == (934, 50, 4), f"Expected (934, 50, 4), got {data.quats.shape}"

        # Check frametime is parsed
        assert data.frametime is not None, "frametime should be parsed"
        assert abs(data.frametime - 1.0 / 60.0) < 1e-6, (
            f"Expected frametime ~0.0167 (60fps), got {data.frametime}"
        )


class TestHCMocapLoading:
    """Test _load_bvh_file() with hc_mocap format."""

    def test_load_bvh_file_hc_mocap_scale(self):
        """Verify hc_mocap loading: meter scale (root_z ≈ 0.84), LeftFootMod present, fps=60."""
        bvh_path = "data/motion_corrected_v2.bvh"
        provider = BVHInputProvider(bvh_path=bvh_path, human_format="hc_mocap")

        # Check fps (should be 60 from BVH, no downsampling in current implementation)
        assert provider.fps == 60, f"Expected fps=60, got {provider.fps}"

        # Check frame count (934 frames, downsampled to 467)
        assert len(provider) == 467, f"Expected 467 frames (934/2), got {len(provider)}"

        # Get first frame
        frame = provider.get_frame()

        # Check LeftFootMod is present
        frame_keys = list(frame.keys())
        assert "LeftFootMod" in frame_keys, "LeftFootMod should be synthesized"
        assert "RightFootMod" in frame_keys, "RightFootMod should be synthesized"

        # Check root_z is in meter scale (≈ 0.84)
        root_data = frame["hc_Abdomen"]
        root_pos = root_data[0]  # First element is position
        root_z = root_pos[2]
        assert 0.5 < root_z < 1.5, f"Expected root_z ≈ 0.84 (meter scale), got {root_z}"


class TestHCMocapProvider:
    """Test BVHInputProvider with hc_mocap format."""

    def test_bvh_input_provider_hc_mocap(self):
        """Verify BVHInputProvider.get_frame() returns valid HumanFrame dict."""
        bvh_path = "data/motion_corrected_v2.bvh"
        provider = BVHInputProvider(bvh_path=bvh_path, human_format="hc_mocap")

        # Get frame 0
        frame = provider.get_frame()

        # Check HumanFrame structure (dict of joint_name -> (pos, quat))
        assert isinstance(frame, dict), "Frame should be a dict"
        assert "hc_Abdomen" in frame, "hc_Abdomen (root) should be present"

        # Check joint data structure: (position, orientation)
        root_data = frame["hc_Abdomen"]
        assert len(root_data) == 2, "Each joint should have (position, orientation)"
        pos, quat = root_data

        # Check quaternion shape
        assert quat.shape == (4,), f"Expected quaternion shape (4,), got {quat.shape}"

        # Check position shape
        assert pos.shape == (3,), f"Expected position shape (3,), got {pos.shape}"
