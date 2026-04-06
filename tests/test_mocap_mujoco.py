from __future__ import annotations

import mujoco
import numpy as np
import pytest

from conftest import requires_mujoco
from teleopit.inputs.pico4_provider import BODY_JOINT_NAMES, BODY_JOINT_PARENTS, Pico4InputProvider
from teleopit.inputs.zmq_provider import ZMQInputProvider
from teleopit.runtime.common import parse_viewers
from teleopit.sim.mocap_mujoco import (
    MocapSkeletonSceneDrawer,
    configure_mocap_camera,
    fit_mocap_camera,
    create_mocap_viewer_model,
    frame_positions_from_human_frame,
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
    assert parse_viewers({"viewers": ["mocap", "retarget"]}) == {"mocap", "retarget"}
    with pytest.raises(ValueError, match="Unsupported viewers"):
        parse_viewers({"viewers": ["bvh", "retarget"]})


def test_pico4_and_zmq_providers_expose_mocap_skeleton_metadata() -> None:
    pico4 = object.__new__(Pico4InputProvider)
    zmq = object.__new__(ZMQInputProvider)

    assert pico4.bone_names == list(BODY_JOINT_NAMES)
    np.testing.assert_array_equal(pico4.bone_parents, BODY_JOINT_PARENTS)
    assert zmq.bone_names == list(BODY_JOINT_NAMES)
    np.testing.assert_array_equal(zmq.bone_parents, BODY_JOINT_PARENTS)
