from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]

_BONE_RGBA = np.array([1.0, 0.55, 0.15, 1.0], dtype=np.float32)
_JOINT_RGBA = np.array([1.0, 0.95, 0.85, 1.0], dtype=np.float32)


def create_mocap_viewer_model(title: str = "Mocap Input") -> mujoco.MjModel:
    xml = f"""
<mujoco model="{title}">
  <visual>
    <global offwidth="1280" offheight="720"/>
    <map znear="0.01" zfar="50"/>
    <headlight ambient="0.15 0.15 0.15" diffuse="0.2 0.2 0.2" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.72 0.72 0.72" width="512" height="512"/>
    <material name="ground_mat" texture="grid" texrepeat="8 8" reflectance="0.15"/>
  </asset>
  <worldbody>
    <light pos="2 -2 5" dir="-0.3 0.3 -1" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>
    <light pos="-2 2 4" dir="0.3 -0.3 -1" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <light pos="0 0 6" dir="0 0 -1" diffuse="0.3 0.3 0.3" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <geom name="ground" type="plane" size="8 8 0.1" material="ground_mat"/>
  </worldbody>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def configure_mocap_camera(camera: mujoco.MjvCamera, root_pos: Sequence[float], distance: float = 2.6) -> None:
    root = np.asarray(root_pos, dtype=np.float64).reshape(3)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.lookat[:] = [root[0], root[1], 0.8]
    camera.distance = distance
    camera.azimuth = 135.0
    camera.elevation = -15.0


def fit_mocap_camera(
    camera: mujoco.MjvCamera,
    positions: Float64Array,
    *,
    min_distance: float = 2.6,
    root_index: int = 0,
    max_root_distance: float = 3.0,
) -> None:
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pos.size == 0:
        configure_mocap_camera(camera, [0.0, 0.0, 0.8], distance=min_distance)
        return

    finite_mask = np.all(np.isfinite(pos), axis=1)
    valid_pos = pos[finite_mask]
    if valid_pos.shape[0] == 0:
        configure_mocap_camera(camera, [0.0, 0.0, 0.8], distance=min_distance)
        return

    if 0 <= root_index < pos.shape[0] and np.all(np.isfinite(pos[root_index])):
        root_pos = pos[root_index]
        root_dist = np.linalg.norm(valid_pos - root_pos.reshape(1, 3), axis=1)
        root_filtered = valid_pos[root_dist <= float(max_root_distance)]
        if root_filtered.shape[0] >= 2:
            valid_pos = root_filtered

    mins = np.min(valid_pos, axis=0)
    maxs = np.max(valid_pos, axis=0)
    center = 0.5 * (mins + maxs)
    extent = np.max(maxs - mins)
    distance = max(float(min_distance), float(extent) * 2.2)

    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.lookat[:] = center
    camera.distance = distance
    camera.azimuth = 135.0
    camera.elevation = -15.0


def frame_positions_from_human_frame(
    bone_names: Sequence[str],
    human_frame: dict[str, tuple[object, object]],
) -> Float64Array:
    positions = np.zeros((len(bone_names), 3), dtype=np.float64)
    for index, bone_name in enumerate(bone_names):
        if bone_name not in human_frame:
            continue
        positions[index] = np.asarray(human_frame[bone_name][0], dtype=np.float64).reshape(3)
    return positions


def compute_ground_lift_offset(positions: Float64Array) -> float:
    """Compute a non-negative Z lift so the lowest joint sits at Z=0."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pos.size == 0:
        return 0.0
    finite_mask = np.all(np.isfinite(pos), axis=1)
    if not np.any(finite_mask):
        return 0.0
    min_z = float(np.min(pos[finite_mask, 2]))
    return max(-min_z, 0.0)


def has_non_degenerate_mocap_positions(
    positions: Float64Array,
    *,
    zero_atol: float = 1e-9,
    min_extent: float = 1e-6,
) -> bool:
    """Return true once shared mocap positions look like a real skeleton frame."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pos.size == 0:
        return False

    finite_mask = np.all(np.isfinite(pos), axis=1)
    valid_pos = pos[finite_mask]
    if valid_pos.shape[0] == 0:
        return False

    nonzero_pos = valid_pos[np.linalg.norm(valid_pos, axis=1) > zero_atol]
    if nonzero_pos.shape[0] < 2:
        return False

    extent = float(np.max(np.ptp(nonzero_pos, axis=0)))
    return extent > min_extent


def resolve_mocap_ground_lift_offset(
    positions: Float64Array,
    current_lift_offset: float | None,
) -> float | None:
    """Initialize a stable mocap ground lift only after receiving a real frame."""
    if current_lift_offset is not None:
        return current_lift_offset
    if has_non_degenerate_mocap_positions(positions):
        return compute_ground_lift_offset(positions)
    return None


def lift_positions_above_ground(
    positions: Float64Array,
    *,
    lift_offset: float | None = None,
) -> Float64Array:
    """Shift all positions up by a fixed lift offset or the current-frame minimum."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    offset = compute_ground_lift_offset(pos) if lift_offset is None else float(lift_offset)
    if offset > 0.0:
        pos = pos.copy()
        pos[:, 2] += offset
    return pos


class MocapSkeletonSceneDrawer:
    def __init__(
        self,
        parents: Sequence[int],
        *,
        joint_radius: float = 0.03,
        bone_radius: float = 0.018,
    ) -> None:
        self._parents = np.asarray(parents, dtype=np.int32).reshape(-1)
        self._joint_radius = float(joint_radius)
        self._bone_radius = float(bone_radius)
        self._num_bones = int(np.sum(self._parents >= 0))

    @property
    def required_geoms(self) -> int:
        return int(self._parents.shape[0] + self._num_bones)

    def draw(self, scene: mujoco.MjvScene, positions: Float64Array) -> None:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.shape != (self._parents.shape[0], 3):
            raise ValueError(
                f"Mocap positions shape mismatch: expected {(self._parents.shape[0], 3)}, got {pos.shape}"
            )

        if scene.ngeom + self.required_geoms > scene.maxgeom:
            raise ValueError(
                f"MuJoCo scene maxgeom too small for mocap skeleton: need {scene.ngeom + self.required_geoms}, have {scene.maxgeom}"
            )

        identity_mat = np.eye(3, dtype=np.float64).reshape(-1)
        sphere_size = np.array([self._joint_radius, 0.0, 0.0], dtype=np.float64)
        capsule_size = np.array([self._bone_radius, 0.0, 0.0], dtype=np.float64)

        for joint_index, parent_index in enumerate(self._parents):
            if parent_index < 0:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=capsule_size,
                pos=np.zeros(3, dtype=np.float64),
                mat=identity_mat,
                rgba=_BONE_RGBA,
            )
            mujoco.mjv_connector(
                geom,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=self._bone_radius,
                from_=pos[parent_index],
                to=pos[joint_index],
            )
            scene.ngeom += 1

        for joint_pos in pos:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=sphere_size,
                pos=joint_pos,
                mat=identity_mat,
                rgba=_JOINT_RGBA,
            )
            scene.ngeom += 1
