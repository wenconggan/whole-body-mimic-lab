import math
import os
import sys
import time
import torch
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
import mujoco
import numpy as np
import mujoco.viewer
import mujoco_viewer
from tqdm import tqdm
from core.config import Config
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
import onnxruntime
import glfw

import onnxruntime
import torch
import numpy as np

def load_onnx_policy(path: str):
    session = onnxruntime.InferenceSession(
        path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    input_names = [inp.name for inp in session.get_inputs()]

    def run_inference(obs_numpy: np.ndarray, time_step_numpy: np.ndarray) -> torch.Tensor:
        ort_inputs = {
            input_names[0]: obs_numpy,       # "obs"
            input_names[1]: time_step_numpy  # "time_step"
        }
        ort_outs = session.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], dtype=torch.float32, device="cuda:0")

    return run_inference


def get_gravity_orientation_from_rpy(roll, pitch):
    rot = R.from_euler('xy', [roll, pitch])
    g_world = np.array([0, 0, -1])
    g_local = rot.inv().apply(g_world)
    return g_local
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    w ,x, y, z= quat
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y,yaw_z])  # , yaw_z


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

simulation_dt = 0.001
control_decimation = 20
num_actions = 18
num_obs = 96
num_history = 6
action_scale = 0.25
ang_vel_scale = 1
dof_pos_scale = 1
dof_vel_scale = 0.05

default_angles = np.array([
    0.0, 0.0, -0.3,  0.6, -0.3,
    0.0, 0.0, -0.3,  0.6, -0.3,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0
], dtype=np.float32)

kps = np.array([200, 200, 200, 200, 30,
                200, 200, 200, 200, 30,
                90, 90, 90, 80,
                90, 90, 90, 80], dtype=np.float32)

kds = np.array([2.5, 2.5, 2.5, 2.0, 1.0,
                2.5, 2.5, 2.5, 2.0, 1.0,
                2.0, 1.0, 1.0, 1.0,
                2.0, 1.0, 1.0, 1.0], dtype=np.float32)

joint_limit_lo = [-0.7854, -0.3490, -2.4434, -0., -1.2217,
                  -0.7854, -0.524, -2.4434, -0, -1.2217,
                  -2, -0.350, -2.618, -1.,
                  -2., -1.57, -2.618, -1.]

joint_limit_hi = [0.7854, 0.5240, 0.5260, 2.3561, 1.2217,
                  0.7854, 0.3490, 0.526, 2.3561, 1.2217,
                  2, 1.570, 2.618, 1.60,
                  2, 0.35, 2.618, 1.60]

torque_limits = [100.0, 100.0, 100.0, 100.0, 100.0,
                 100.0, 100.0, 100.0, 100.0, 100.0,
                 30.0, 30.0, 30.0, 30.0,
                 30.0, 30.0, 30.0, 30.0]

xml_path = f'robots/x218dof/x2.xml'
soft_dof_pos_limit = 0.98
motion_start_times = torch.zeros(1, dtype=torch.float32, device="cpu")


policy_path = ("/home/wenconggan/whole_body_tracking/logs/rsl_rl/x2_flat/2025-09-01_11-17-33/exported/policy.onnx")
# policy_path = "/home/wenconggan/mimic/logs/TEST/20250901_143937-TEST_Motion_Tracking-motion_tracking-x2/exported/model_16000.onnx"


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

class Sim2Sim():
    def __init__(self, _cfg, ):
        self.sign_mask = np.ones(18)
        indices_to_flip = [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17]  # 你可以改成你自己的目标索引
        self.sign_mask[indices_to_flip] = -1
        self.cfg = _cfg
        # self.policy = _policy
        self.target_q = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        self.action = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        self.action_filter = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        self.hist_obs = deque()
        for _ in range(5):
            self.hist_obs.append(np.zeros([1, 61], dtype=np.double))
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"Joint {i}: {name}")

        self.data = mujoco.MjData(self.model)


    def get_obs(self, ):

        q = self.data.qpos[7:]
        dq = self.data.qvel[6:]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]
        euler = quaternion_to_euler_array(quat)
        # euler[euler > math.pi] -= 2 * math.pi
        g_local = get_gravity_orientation_from_rpy(euler[0], euler[1])
        gravity_orientation = g_local
        return q, dq, omega, euler,gravity_orientation

    def set_sim_target(self, target_q):
        q = self.data.qpos.astype(np.double)[7:]
        dq = self.data.qvel.astype(np.double)[6:]
        tau = (target_q - q) * kps - dq * kds
        self.data.ctrl = tau
        mujoco.mj_step(self.model, self.data)

    def run(self):

        policy = load_onnx_policy(policy_path)
        action = np.zeros(num_actions, dtype=np.float32)
        obs = np.zeros(num_obs , dtype=np.float32)
        counter = 0

        while True:
            pbar = tqdm(range(int(self.cfg.env.run_duration / 0.001)), desc="Simulating")
            m = mujoco.MjModel.from_xml_path(xml_path)
            d = mujoco.MjData(m)
            m.opt.timestep = simulation_dt
            viewer = mujoco_viewer.MujocoViewer(m, d, width=2000, height=1000)
            glfw.set_window_pos(viewer.window, 1000, 100)  # x, y
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat = [0, 0, 1]
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            self.target_q = default_angles*self.sign_mask

            d.qpos[7:] = default_angles*self.sign_mask
            mujoco.mj_step(m, d)
            for _ in pbar:

                    tau = pd_control(self.target_q, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                    d.ctrl[:] = tau
                    mujoco.mj_step(m, d)

                    counter += 1

                    if counter % control_decimation == 0:
                        qj = (d.qpos[7:] - default_angles*self.sign_mask) * dof_pos_scale * self.sign_mask
                        dqj = d.qvel[6:]  * dof_vel_scale * self.sign_mask
                        omega = d.qvel[3:6] * ang_vel_scale
                        quat = d.qpos[3:7]
                        euler = quaternion_to_euler_array(quat)
                        g_local = get_gravity_orientation_from_rpy(euler[0], euler[1])
                        gravity_orientation = g_local

                        obs[:num_actions] = 0
                        obs[num_actions:num_actions * 2] = 0
                        obs[num_actions * 2 : num_actions * 2 +3] = gravity_orientation
                        obs[num_actions * 2 + 3:num_actions * 2 + 3 + 3] = omega
                        obs[num_actions * 2 + 3 + 3:num_actions * 3 + 3 + 3] = qj
                        obs[num_actions * 4 + 3 + 3:] = dqj

                        # 假设 time_step 是从 counter 或 step 计算出来
                        time_step_val = np.array([[counter]], dtype=np.float32)

                        obs_tensor = torch.from_numpy(obs).unsqueeze(0).cpu().numpy()
                        action = policy(obs_tensor, time_step_val).cpu().numpy().squeeze()


                        # obs_tensor = torch.from_numpy(obs).unsqueeze(0).reshape(1, -1).cpu().numpy()
                        # action = policy(obs_tensor).detach().cpu().numpy().squeeze()
                        self.action = action * self.sign_mask
                        self.target_q = self.action * action_scale + default_angles*self.sign_mask
                    viewer.render()

if __name__ == '__main__':
    # model_path = "../models/policy_1.pt"
    # policy = torch.jit.load(model_path)
    mybot = Sim2Sim(Config)
    mybot.run()

