import argparse
import os
import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import onnxruntime
import onnx
import numpy as np
import mujoco
import yaml
from scipy.spatial.transform import Rotation as R
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
xml_path = os.path.join(project_root, "source", "whole_body_tracking", "whole_body_tracking", "assets", "x2", "x2.xml")

mujoco_joint_index = [
    "L_hip_yaw",
    "L_hip_roll",
    "L_hip_pitch",
    "L_knee_pitch",
    "L_ankle_pitch",
    "R_hip_yaw",
    "R_hip_roll",
    "R_hip_pitch",
    "R_knee_pitch",
    "R_ankle_pitch",
    "L_shoulder_pitch",
    "L_shoulder_roll",
    "L_shoulder_yaw",
    "L_elbow_pitch",
    "R_shoulder_pitch",
    "R_shoulder_roll",
    "R_shoulder_yaw",
    "R_elbow_pitch",
]

def get_gravity_orientation_from_rpy(roll, pitch):
    rot = R.from_euler('xy', [roll, pitch])
    g_world = np.array([0, 0, -1])
    g_local = rot.inv().apply(g_world)
    return g_local

def quaternion_to_euler_array(quat):
    w ,x, y, z= quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y,yaw_z])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_to_euler(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
    if q_vec.ndim == 2:
        dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)
        c = q_vec * dot_product * 2.0
    else:
        dot_product = np.expand_dims(np.einsum('...i,...i->...', q_vec, v), axis=-1)
        c = q_vec * dot_product * 2.0
    return a - b + c

def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b):
    rotm_a = np.zeros(9)
    mujoco.mju_quat2Mat(rotm_a, quat_a)
    rotm_a = rotm_a.reshape(3, 3)
    rel_pos = rotm_a.T @ (pos_b - pos_a)
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), quat_b)
    rel_quat = rel_quat / np.linalg.norm(rel_quat)
    return rel_pos, rel_quat

def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, required=True, help="动作文件路径")
    parser.add_argument("--policy_path", type=str, required=True, help="ONNX 策略文件路径")
    args = parser.parse_args()
    motion_file = args.motion_file
    policy_path = args.policy_path
    print(f"使用策略onnx模型: {policy_path}")
    print(f"使用动作npz 文件: {motion_file}")
    # motion_file = "../../motion/chars.npz"
    # policy_path ="/home/wenconggan/whole_body_tracking/logs/rsl_rl/x2_flat/2025-09-01_11-17-33/exported/policy.onnx"
    model = onnx.load(policy_path)
    joint_names = []
    joint_pos_seq = []
    stiffness_seq = []
    damping_seq = []
    action_scale_seq = []
    for prop in model.metadata_props:
        values = prop.value.split(",")
        if prop.key == "joint_names":
            joint_names = values
        elif prop.key == "default_joint_pos":
            joint_pos_seq = np.array([float(x) for x in values])
        elif prop.key == "joint_stiffness":
            stiffness_seq = np.array([float(x) for x in values])
        elif prop.key == "joint_damping":
            damping_seq = np.array([float(x) for x in values])
        elif prop.key == "action_scale":
            action_scale_seq = np.array([float(x) for x in values])

    joint_pos_array = np.array([joint_pos_seq[joint_names.index(joint)] for joint in mujoco_joint_index])
    stiffness_array = np.array([stiffness_seq[joint_names.index(joint)] for joint in mujoco_joint_index])
    damping_array = np.array([damping_seq[joint_names.index(joint)] for joint in mujoco_joint_index])
    action_scale_array = np.array([action_scale_seq[joint_names.index(joint)] for joint in mujoco_joint_index])

    print(f"{'Joint Name':20} {'Pos':>8} {'Stiffness':>10} {'Damping':>8} {'ActionScale':>12}")
    print("-" * 60)
    for i, joint in enumerate(mujoco_joint_index):
        print(
            f"{joint:20} {joint_pos_array[i]:8.3f} {stiffness_array[i]:10.3f} {damping_array[i]:8.3f} {action_scale_array[i]:12.3f}")

    for prop in model.metadata_props:
        if prop.key not in ["joint_names", "default_joint_pos", "joint_stiffness", "joint_damping", "action_scale"]:
            print(f"{prop.key}: {prop.value}")

    motion =  np.load(motion_file)
    motion_body_pos = motion["body_pos_w"]
    motion_body_quat = motion["body_quat_w"]
    motion_joint_pos = motion["joint_pos"]
    motion_joint_vel = motion["joint_vel"]
    simulation_duration = 1000.0
    simulation_dt = 0.001
    counter = 0
    timestep = 0
    control_decimation = 20
    num_actions = 18
    num_obs = 96

    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    policy = onnxruntime.InferenceSession(policy_path)
    input_name = policy.get_inputs()[0].name
    output_name = policy.get_outputs()[0].name
    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    motion_body_pos_cur = motion_body_pos[timestep,9,:]
    motion_body_quat_cur = motion_body_quat[timestep,9,:]
    target_dof_pos = joint_pos_array.copy()
    d.qpos[7:] = target_dof_pos
    body_name = "pelvis_link"
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            max_timestep = motion_joint_pos.shape[0]
            if timestep >= max_timestep:
                timestep = 0
            frame_idx = timestep % motion_joint_pos.shape[0]
            print(f"\rTimestep: {timestep} / {motion_joint_pos.shape[0]} (frame_idx: {frame_idx})", end="")

            mujoco.mj_step(m, d)
            tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)# xml

            d.ctrl[:] = tau
            counter += 1
            if counter % control_decimation == 0:

                position = d.xpos[body_id]
                quaternion = d.xquat[body_id]
                euler = quaternion_to_euler_array(quaternion)
                g_local = get_gravity_orientation_from_rpy(euler[0], euler[1])
                command = np.concatenate((motion_joint_pos[timestep,:],motion_joint_vel[timestep,:]),axis=0)
                motion_body_pos_cur = motion_body_pos[timestep,9,:]
                motion_body_quat_cur = motion_body_quat[timestep,9,:]
                anchor_quat = subtract_frame_transforms_mujoco(position,quaternion,motion_body_pos_cur,motion_body_quat_cur)[1]
                anchor_ori = np.zeros(9)
                mujoco.mju_quat2Mat(anchor_ori, anchor_quat)
                anchor_ori = anchor_ori.reshape(3, 3)[:, :2]
                anchor_ori = anchor_ori.reshape(-1,)
                joint_pos = np.array([d.qpos[7: 7 + num_actions][mujoco_joint_index.index(joint)] for joint in joint_names])
                joint_vel = np.array([d.qvel[6: 6 + num_actions][mujoco_joint_index.index(joint)] for joint in joint_names])

                obs[0: 36] = command
                obs[36:36 +  3] = g_local
                obs[39:39 +  3] = d.qvel[3 : 6]
                obs[42:42 + 18] = joint_pos - joint_pos_seq
                obs[60:60 + 18] = joint_vel
                obs[78:78 + 18] = action_buffer
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy.run(['actions'], {'obs': obs_tensor.numpy(),'time_step':np.array([timestep], dtype=np.float32).reshape(1,1)})[0]
                action = np.asarray(action).reshape(-1)
                action_buffer = action.copy()
                target_dof_pos = action * action_scale_seq + joint_pos_seq
                target_dof_pos = target_dof_pos.reshape(-1,)
                target_dof_pos = np.array([target_dof_pos[joint_names.index(joint)] for joint in mujoco_joint_index])
                timestep+=1

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
