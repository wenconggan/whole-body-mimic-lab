import numpy as np
from itertools import chain


class Config:
    class env:
        # change the observation dim
        frame_stack = 1
        num_single_obs = 99
        num_observations = int(frame_stack * num_single_obs)
        num_actions = 18
        run_duration = 50000.0  # 单位s
        num_arm_actions = 10
        num_leg_actions = 10
        grpc_channel = '192.168.55.204'  # r6s


    class sim_config:
        # mujoco_model_path = f'../robots/X02Lite/X02Lite.xml'
        dt = 0.005
        action_scale = 0.25
        decimation = 10
        cycle_time = 0.64
        mujoco_model_path = f'/home/wenconggan/mimic/deploy/robots/x218dof/x2_vis.xml'
        # mujoco_model_path = f'/home/wenconggan/视频/x2mimic/loco_deploy/robots/x2/x2.xml'

        pos0 = np.array([0, 0.0, -0.3, 0.6, -0.3, 0, 0.0, -0.3, 0.6, -0.3,
                        -0.0, 0.0, 0, 0.0,
                        -0.0, 0.0, 0, 0.0], dtype=np.double)
        kps = 1 * np.array([200, 200, 200, 200, 50, 200, 200, 200, 200, 50,100,100,100,100,100,100,100,100], dtype=np.double)
        kds = np.array([2, 2, 2, 4, 2, 2, 2, 2, 4, 2,                   2,2, 2, 2, 2, 2, 2, 2], dtype=np.double)
        tau_limit = 50000. * np.ones(18, dtype=np.double)

    class real_config:
        dt = 0.001
        action_scale = 0.25
        decimation = 10
        cycle_time = 0.8
        standkps = np.array([120, 200, 200, 200, 100, 120, 200, 200, 200, 100], dtype=np.double)
        standkds = np.array([1, 5, 5, 5, 2, 1, 5, 5, 5, 2], dtype=np.double)
        standpos0 = np.array([
                        0, 0.05, 0.3, -0.6, 0.3,
                        0, 0.05, 0.3, -0.6, 0.3,
                        -0.0, 0.0, 0, 0,
                        -0.0, 0.0, 0, 0,  ], dtype=np.double)
        pos0 = np.array([
                        0, 0.00, 0.3, -0.6, 0.3,
                        0, 0.00, 0.3, -0.6, 0.3,
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,   ], dtype=np.double)

        turn_pos0 = np.array([
                        0, 0.00, 0.3, -0.6, 0.3,
                        0, 0.00, 0.3, -0.6, 0.3,
                        -0, 0, 0, 0,
                        -0, 0, 0, 0], dtype=np.double)

        leg_kps = np.array([100, 200, 220, 220, 30,  100, 200, 220, 220, 30], dtype=np.double) * 1.0
        leg_kds = np.array([  2,   3,   3,   3,  2,    2,   3,   3,   3,  2], dtype=np.double) * 1.0
        tau_limit = 500. * np.ones(20, dtype=np.double)

    class control:
        action_scale = 0.25
        decimation = 10
        cycle_time = 0.60

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1
            dof_pos = 1.
            dof_vel = 0.05
        clip_observations = 100.

    class robot_config:
        clip_actions_upper = np.array([1.04, 0.34, 2.00, 0.08, 1.22, 1.04, 0.34, 2.00, 0.08, 1.22], dtype=np.double)
        clip_actions_lower = np.array([-1.04, -0.34, -0.35, -2.44, -1.22, -1.04, -0.34, -0.35, -2.44, -1.22], dtype=np.double)
        tau_limit = 50000. * np.ones(10, dtype=np.double)

