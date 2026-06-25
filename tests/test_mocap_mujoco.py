from __future__ import annotations

import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from conftest import requires_mujoco
from teleopit.inputs.pico4_provider import BODY_JOINT_NAMES, BODY_JOINT_PARENTS, Pico4InputProvider
from teleopit.runtime.common import parse_viewers
from teleopit.sim.mocap_mujoco import (
    MocapSkeletonSceneDrawer,
    configure_mocap_camera,
    compute_ground_lift_offset,
    fit_mocap_camera,
    create_mocap_viewer_model,
    frame_positions_from_human_frame,
    has_non_degenerate_mocap_positions,
    resolve_mocap_ground_lift_offset,
    lift_positions_above_ground,
)


@requires_mujoco
def test_frame_positions_from_human_frame_preserves_bone_order() -> None:
    positions = frame_positions_from_human_frame(
        ["root", "hip", "head"],
        {
            "head": (np.array([3.0, 4.0, 5.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            "root": (np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        },
    )

    np.testing.assert_allclose(
        positions,
        np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 5.0],
            ],
            dtype=np.float64,
        ),
    )


def test_mocap_ground_lift_waits_for_real_frame() -> None:
    zero_frame = np.zeros((4, 3), dtype=np.float64)
    actual_frame = np.array(
        [
            [0.0, 0.0, -1.2],
            [0.2, 0.0, -0.8],
            [0.4, 0.1, -0.5],
            [0.6, 0.0, -0.2],
        ],
        dtype=np.float64,
    )

    assert not has_non_degenerate_mocap_positions(zero_frame)
    assert resolve_mocap_ground_lift_offset(zero_frame, None) is None

    assert has_non_degenerate_mocap_positions(actual_frame)
    offset = resolve_mocap_ground_lift_offset(actual_frame, None)
    assert offset == pytest.approx(1.2)
    np.testing.assert_allclose(
        lift_positions_above_ground(actual_frame, lift_offset=offset),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.4],
                [0.4, 0.1, 0.7],
                [0.6, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )

    assert resolve_mocap_ground_lift_offset(actual_frame, offset) == offset
    assert compute_ground_lift_offset(actual_frame) == pytest.approx(1.2)


@requires_mujoco
def test_mocap_scene_drawer_populates_custom_geoms() -> None:
    model = create_mocap_viewer_model()
    scene = mujoco.MjvScene(model, maxgeom=32)
    drawer = MocapSkeletonSceneDrawer([-1, 0, 1])

    drawer.draw(
        scene,
        np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.2],
                [0.2, 0.0, 1.4],
            ],
            dtype=np.float64,
        ),
    )

    assert scene.ngeom == drawer.required_geoms


@requires_mujoco
def test_configure_mocap_camera_tracks_root_position() -> None:
    camera = mujoco.MjvCamera()

    configure_mocap_camera(camera, [1.5, -2.0, 0.3], distance=3.2)

    np.testing.assert_allclose(camera.lookat, np.array([1.5, -2.0, 0.8], dtype=np.float64))
    assert camera.distance == 3.2


@requires_mujoco
def test_fit_mocap_camera_tracks_full_skeleton_extent() -> None:
    camera = mujoco.MjvCamera()
    positions = np.array(
        [
            [10.0, 0.0, 100.0],
            [20.0, 5.0, 110.0],
            [30.0, 10.0, 120.0],
        ],
        dtype=np.float64,
    )

    fit_mocap_camera(camera, positions, min_distance=2.6)

    np.testing.assert_allclose(camera.lookat, np.array([20.0, 5.0, 110.0], dtype=np.float64))
    assert camera.distance > 2.6


@requires_mujoco
def test_fit_mocap_camera_ignores_far_outlier_from_root() -> None:
    camera = mujoco.MjvCamera()
    positions = np.array(
        [
            [1.0, 2.0, 0.9],
            [1.2, 2.1, 1.7],
            [50.0, 60.0, 70.0],
        ],
        dtype=np.float64,
    )

    fit_mocap_camera(camera, positions, min_distance=2.6)

    np.testing.assert_allclose(camera.lookat, np.array([1.1, 2.05, 1.3], dtype=np.float64))
    assert camera.distance == pytest.approx(2.6)


def test_parse_viewers_accepts_mocap_and_rejects_bvh() -> None:
    assert parse_viewers({"viewers": ["mocap", "retarget", "camera"]}) == {"mocap", "retarget", "camera"}
    assert parse_viewers({"viewers": "all"}) == {"mocap", "retarget", "sim2sim"}
    with pytest.raises(ValueError, match="Unsupported viewers"):
        parse_viewers({"viewers": ["bvh", "retarget"]})


def test_pico4_provider_exposes_mocap_skeleton_metadata() -> None:
    pico4 = object.__new__(Pico4InputProvider)

    assert pico4.bone_names == list(BODY_JOINT_NAMES)
    np.testing.assert_array_equal(pico4.bone_parents, BODY_JOINT_PARENTS)


def test_pico4_provider_exposes_controller_snapshot() -> None:
    pico4 = object.__new__(Pico4InputProvider)
    pico4._lock = threading.Lock()
    pico4._controller_snapshot = None
    pico4._last_source_seq = None

    frame = SimpleNamespace(
        seq=42,
        controllers=SimpleNamespace(
            left=SimpleNamespace(raw=True, axis={"grip": 0.75, "trigger": 0.25}),
            right=SimpleNamespace(raw=False, axis={}),
        ),
    )
    pico4._accept_controller_snapshot(frame, timestamp=12.5)

    snapshot = pico4.get_controller_snapshot()
    assert snapshot is not None
    assert snapshot.seq == 42
    assert snapshot.timestamp_s == pytest.approx(12.5)
    assert snapshot.left.raw is True
    assert snapshot.left.grip == pytest.approx(0.75)
    assert snapshot.left.trigger == pytest.approx(0.25)
    assert snapshot.right.raw is False
