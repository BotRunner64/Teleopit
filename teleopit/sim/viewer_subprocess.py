"""Subprocess viewer functions for MuJoCo robot and mocap visualization.

Each function runs in its own process with a GLFW context.
"""

from __future__ import annotations

import multiprocessing as mp
import time


def robot_viewer_proc(
    xml_path: str,
    qpos_arr: mp.Array,
    qpos_len: int,
    shutdown: mp.Event,
    alive: mp.Value,
    foot_z_correction: bool,
    left_foot_name: str,
    right_foot_name: str,
    title: str = "",
    win_x: int = -1,
    win_y: int = -1,
) -> None:
    """Subprocess: robot model viewer — displays qpos with optional foot Z fix.

    Used for both sim2sim (physics result) and retarget (kinematic result).
    """
    import mujoco
    import mujoco.viewer
    import numpy as np
    import os
    import re

    # Set window title via model name and position via GLFW hints
    if title:
        with open(xml_path) as f:
            xml_str = f.read()
        xml_str = re.sub(r'<mujoco\s+model="[^"]*"', f'<mujoco model="{title}"', xml_str)
        os.chdir(os.path.dirname(os.path.abspath(xml_path)))
        model = mujoco.MjModel.from_xml_string(xml_str)
    else:
        model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    left_foot_id = -1
    right_foot_id = -1
    if foot_z_correction:
        left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_foot_name)
        right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_foot_name)

    pelvis_id = -1
    try:
        pelvis_id = model.body("pelvis").id
    except Exception:
        pass

    # Set initial window position via GLFW hints (GLFW 3.4+)
    if win_x >= 0 and win_y >= 0:
        try:
            import glfw
            glfw.init()
            glfw.window_hint(glfw.POSITION_X, win_x)
            glfw.window_hint(glfw.POSITION_Y, win_y)
        except Exception:
            pass

    v = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
    v.cam.distance = 2.0
    alive.value = 1

    try:
        while v.is_running() and not shutdown.is_set():
            with qpos_arr.get_lock():
                qpos = np.array(qpos_arr[:qpos_len], dtype=np.float64)

            data.qpos[:qpos_len] = qpos
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            if foot_z_correction and left_foot_id >= 0 and right_foot_id >= 0:
                lowest_z = min(data.xpos[left_foot_id][2], data.xpos[right_foot_id][2])
                if lowest_z < 0.0:
                    data.qpos[2] -= lowest_z
                    mujoco.mj_forward(model, data)

            if pelvis_id >= 0:
                v.cam.lookat[:] = data.xpos[pelvis_id]
            else:
                v.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.8]
            v.sync()
            time.sleep(0.02)
    finally:
        alive.value = 0
        try:
            v.close()
        except Exception:
            pass


def mocap_viewer_proc(
    parents_list: list[int],
    pos_arr: mp.Array,
    n_bones: int,
    shutdown: mp.Event,
    alive: mp.Value,
    win_x: int = -1,
    win_y: int = -1,
) -> None:
    """Subprocess: mocap input viewer rendered with MuJoCo custom geoms."""
    import mujoco
    import mujoco.viewer
    import numpy as np

    from teleopit.sim.mocap_mujoco import (
        MocapSkeletonSceneDrawer,
        create_mocap_viewer_model,
        fit_mocap_camera,
        lift_positions_above_ground,
        resolve_mocap_ground_lift_offset,
    )

    model = create_mocap_viewer_model()
    data = mujoco.MjData(model)
    drawer = MocapSkeletonSceneDrawer(parents_list)

    if win_x >= 0 and win_y >= 0:
        try:
            import glfw
            glfw.init()
            glfw.window_hint(glfw.POSITION_X, win_x)
            glfw.window_hint(glfw.POSITION_Y, win_y)
        except Exception:
            pass

    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
    alive.value = 1
    ground_lift_offset: float | None = None

    try:
        while viewer.is_running() and not shutdown.is_set():
            with pos_arr.get_lock():
                pos = np.array(pos_arr[:n_bones * 3], dtype=np.float64).reshape(n_bones, 3)
            ground_lift_offset = resolve_mocap_ground_lift_offset(pos, ground_lift_offset)
            pos = lift_positions_above_ground(pos, lift_offset=ground_lift_offset)

            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.user_scn.ngeom = 0
            drawer.draw(viewer.user_scn, pos)
            fit_mocap_camera(viewer.cam, pos)
            viewer.sync()
            time.sleep(0.03)
    finally:
        alive.value = 0
        try:
            viewer.close()
        except Exception:
            pass


def start_robot_viewer(
    xml_path: str, nq: int, foot_z_correction: bool,
    title: str = "", win_x: int = -1, win_y: int = -1,
) -> tuple[mp.Process, mp.Array, mp.Value, mp.Event]:
    """Launch a subprocess viewer for a robot model.

    Returns (process, qpos_shared_array, alive_flag, shutdown_event).
    """
    arr = mp.Array("d", nq)
    shutdown = mp.Event()
    alive = mp.Value("i", 0)
    proc = mp.Process(
        target=robot_viewer_proc,
        args=(xml_path, arr, nq, shutdown, alive,
              foot_z_correction, "left_ankle_roll_link", "right_ankle_roll_link",
              title, win_x, win_y),
        daemon=True,
    )
    proc.start()
    return proc, arr, alive, shutdown
