#!/usr/bin/env python3
"""Test if ONNX policy produces identical output on different platforms.

Run on both PC and NX with the same policy file and compare output.
Uses fixed synthetic input to eliminate sensor differences.

Usage:
    python scripts/test_policy_determinism.py --policy track.onnx
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SDK_PATH = _REPO_ROOT / "third_party" / "unitree_sdk2_python"
if _SDK_PATH.exists():
    sys.path.insert(0, str(_SDK_PATH))

import numpy as np
import onnxruntime as ort
import mujoco

# Reuse constants from standalone_standing
NUM_JOINTS = 29
MJCF_PATH = _REPO_ROOT / "teleopit" / "retargeting" / "gmr" / "assets" / "unitree_g1" / "g1_mjlab.xml"

DEFAULT_ANGLES = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0,
    -0.312, 0, 0, 0.669, -0.363, 0,
    0, 0, 0,
    0.2, 0.2, 0, 0.6, 0, 0, 0,
    0.2, -0.2, 0, 0.6, 0, 0, 0,
], dtype=np.float32)

ACTION_SCALE = np.array([
    0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
    0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
    0.5475, 0.4386, 0.4386,
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
], dtype=np.float32)


def quat_inv(q):
    inv = q.copy(); inv[..., 1:] = -inv[..., 1:]; return inv

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
    w2, x2, y2, z2 = q2[...,0], q2[...,1], q2[...,2], q2[...,3]
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], axis=-1).astype(np.float32)

def quat_rotate(q, v):
    v_quat = np.zeros(4, dtype=np.float32); v_quat[1:4] = v
    result = quat_mul(quat_mul(q, v_quat), quat_inv(q))
    return result[1:4]

def quat_to_rot6d(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*y+w*z),
                     1-2*(x*x+z*z), 2*(x*z-w*y), 2*(y*z+w*x)], dtype=np.float32)


def build_obs(mj_model, mj_data, anchor_body_id, qpos, qvel, quat, ang_vel, motion_qpos, last_action):
    """Build 166D observation with fixed inputs."""
    # Robot FK
    mj_data.qpos[:] = 0.0
    mj_data.qpos[0:3] = 0.0
    q = np.asarray(quat, dtype=np.float64); q /= max(np.linalg.norm(q), 1e-8)
    mj_data.qpos[3:7] = q
    mj_data.qpos[7:36] = qpos.astype(np.float64)
    mujoco.mj_kinematics(mj_model, mj_data)
    robot_anchor_quat = np.asarray(mj_data.xquat[anchor_body_id], dtype=np.float32).copy()

    # Motion FK
    motion = np.asarray(motion_qpos, dtype=np.float32)
    mj_data.qpos[:] = 0.0
    mj_data.qpos[0:3] = motion[0:3].astype(np.float64)
    mq = motion[3:7].astype(np.float64); mq /= max(np.linalg.norm(mq), 1e-8)
    mj_data.qpos[3:7] = mq
    mj_data.qpos[7:36] = motion[7:36].astype(np.float64)
    mujoco.mj_kinematics(mj_model, mj_data)
    motion_anchor_quat = np.asarray(mj_data.xquat[anchor_body_id], dtype=np.float32).copy()

    # Build obs
    motion_joint_pos = motion[7:36]
    command = np.concatenate((motion_joint_pos, np.zeros(NUM_JOINTS, dtype=np.float32)))
    rel_quat = quat_mul(quat_inv(robot_anchor_quat), motion_anchor_quat)
    ori_b = quat_to_rot6d(rel_quat)
    joint_pos_rel = qpos - DEFAULT_ANGLES

    base_obs = np.concatenate([command, ori_b, ang_vel, joint_pos_rel, qvel, last_action])
    proj_grav = quat_rotate(quat_inv(quat), np.array([0,0,-1], dtype=np.float32))
    ref_proj_grav = quat_rotate(quat_inv(motion_anchor_quat), np.array([0,0,-1], dtype=np.float32))
    velcmd_obs = np.concatenate([proj_grav, np.zeros(3, dtype=np.float32),
                                  np.zeros(3, dtype=np.float32), ref_proj_grav])
    return np.concatenate([base_obs, velcmd_obs]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()

    # Load model
    mj_model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    mj_data = mujoco.MjData(mj_model)
    body_names = {mj_model.body(i).name: i for i in range(mj_model.nbody)}
    anchor_id = body_names["torso_link"]

    # Load policy
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(args.policy, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    multi_input = len(session.get_inputs()) == 2 and session.get_inputs()[1].name == "obs_history"
    if multi_input:
        history_len = int(session.get_inputs()[1].shape[1])

    print(f"Platform: {sys.platform}")
    print(f"Providers: {ort.get_available_providers()}")
    print(f"Using: CPUExecutionProvider (forced)")
    print(f"Multi-input: {multi_input}, history_len: {history_len if multi_input else 0}")
    print(f"MuJoCo: {mujoco.__version__}, numpy: {np.__version__}, ort: {ort.__version__}")
    print()

    # Fixed synthetic inputs (deterministic)
    np.random.seed(42)
    quat = np.array([0.822, -0.005, 0.003, -0.569], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    qpos = DEFAULT_ANGLES + np.random.randn(NUM_JOINTS).astype(np.float32) * 0.02
    qvel = np.random.randn(NUM_JOINTS).astype(np.float32) * 0.01
    ang_vel = np.array([0.001, -0.002, 0.001], dtype=np.float32)
    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)

    # Standing reference
    motion_qpos = np.zeros(36, dtype=np.float32)
    motion_qpos[3] = 1.0
    motion_qpos[7:36] = DEFAULT_ANGLES

    # Build obs
    obs = build_obs(mj_model, mj_data, anchor_id, qpos, qvel, quat, ang_vel, motion_qpos, last_action)
    print(f"obs shape: {obs.shape}")
    print(f"obs[:10]:  {np.array2string(obs[:10], precision=8, separator=',')}")
    print(f"obs[-10:]: {np.array2string(obs[-10:], precision=8, separator=',')}")
    print(f"obs sum:   {obs.sum():.8f}")
    print(f"obs hash:  {hash(obs.tobytes())}")
    print()

    # Run policy 5 steps, accumulating history
    history_buf = []
    for step in range(5):
        obs_batch = obs[np.newaxis, :]
        if multi_input:
            if len(history_buf) == 0:
                history_buf = [obs.copy()] * history_len
            else:
                history_buf.append(obs.copy())
                if len(history_buf) > history_len:
                    history_buf = history_buf[-history_len:]
            obs_history = np.stack(history_buf, axis=0)[np.newaxis].astype(np.float32)
            feed = {input_name: obs_batch, "obs_history": obs_history}
        else:
            feed = {input_name: obs_batch}

        action = session.run([output_name], feed)[0].reshape(-1).astype(np.float32)
        target = np.clip(action, -10, 10) * ACTION_SCALE + DEFAULT_ANGLES

        print(f"Step {step}: action[:6]={np.array2string(action[:6], precision=8, separator=',')}  "
              f"target[:6]={np.array2string(target[:6], precision=8, separator=',')}")

        # Update obs with new last_action for next step
        last_action = action.copy()
        obs = build_obs(mj_model, mj_data, anchor_id, qpos, qvel, quat, ang_vel, motion_qpos, last_action)

    print(f"\nFinal obs sum: {obs.sum():.8f}")
    print(f"Final action sum: {action.sum():.8f}")


if __name__ == "__main__":
    main()
