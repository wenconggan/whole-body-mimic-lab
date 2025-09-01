
import time
import math
import numpy as np


class NanoSleep:
    def __init__(self, ms):
        self.duration_sec = ms * 0.001  # 转化为单位秒

    def waiting(self, _start_time):
        while True:
            current_time = time.perf_counter()
            elapsed_time = current_time - _start_time
            if elapsed_time >= self.duration_sec:
                break


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
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


def init_command(command, num_actions):
    for idx in range(num_actions):
        command.mode.append(1)
        command.position.append(0.0)
        command.velocity.append(0.0)
        command.torque.append(0.0)
        command.ens.append(1)
        command.kp.append(0.1)
        command.kd.append(0.1)
        command.max_torque.append(1)


def set_motor_mode(command, config):
    idx_max = len(config.joint_name)
    command.cmd_enable = 1
    for idx in range(idx_max):
        command.kp[idx] = config.kp[idx]
        command.kd[idx] = config.kd[idx]
        command.max_torque[idx] = config.imax[idx]


# def ref_trajectory(_cfg, cnt_pd_loop):
#     leg_l = math.sin(2 * math.pi * cnt_pd_loop * 0.001 / _cfg.env.cycle_time)  # x * 0.001, ms -> s
#     leg_r = math.sin(2 * math.pi * cnt_pd_loop * 0.001 / _cfg.env.cycle_time)  # x * 0.001, ms -> s
#     ref_dof_pos = np.zeros(self.num_actions, dtype=np.float32)
#     ref_dof_pos[2] = _cfg.robot_config.pos0[2] + leg_l * 0.17
#     ref_dof_pos[3] = _cfg.robot_config.pos0[3] - leg_l * 0.34
#     ref_dof_pos[4] = _cfg.robot_config.pos0[4] + leg_l * 0.17
#
#     ref_dof_pos[7] = _cfg.robot_config.pos0[7] + leg_r * 0.17
#     ref_dof_pos[8] = _cfg.robot_config.pos0[8] - leg_r * 0.34
#     ref_dof_pos[9] = _cfg.robot_config.pos0[9] + leg_r * 0.17
#     return ref_dof_pos


def print_configs(config):
    idx_max = len(config.joint_name)
    line = '-' * idx_max * 11
    print("---------+" + line)
    print("MtrName  |", end="")
    for idx in range(idx_max):
        print(f"{config.joint_name[idx]:>11}", end="")
    print()
    print("---------+" + line)
    for attr in ["pzero", "pmin", "pmax", "imax", "kp", "kd"]:
        print("{:<8} |".format(attr.upper()), end="")
        for i in range(idx_max):
            print("{:>11.3f}".format(getattr(config, attr)[i]), end="")
        print()

    print("---------+" + line)


def print_state(state, config):
    idx_max = len(config.joint_name)
    line = '-' * idx_max * 11
    print(f"system tic: {state.system_tic} ms")
    print("---------+" + line)
    print("MtrName  |", end="")
    for i in range(idx_max):
        print(f"{config.joint_name[i]:>11}", end="")
    print()
    print("---------+" + line)

    labels = ["qc", "dqc", "tqc", "temp", "absc", "loss"]
    for label in labels:
        print(f"{label:<8} |", end="")
        for i in range(idx_max):
            if label == "qc":
                print(f"{state.position[i]:>11.3f}", end="")
            elif label == "dqc":
                print(f"{state.velocity[i]:>11.3f}", end="")
            elif label == "tqc":
                print(f"{state.torque[i]:>11.3f}", end="")
            elif label == "temp":
                print(f"{state.temperature[i]:>11.3f}", end="")
            elif label == "absc":
                print(f"{state.abs_encoder[i]:>11.3f}", end="")
            elif label == "loss":
                print(f"{state.pack_loss[i]:>11}", end="")
        print()
    print("---------+" + line)
    line = '-' * 80
    print(
        f"Foot Sensor (L L R R):      {state.foot_force[0]:>10.3f} {state.foot_force[1]:>10.3f} {state.foot_force[2]:>15.3f} {state.foot_force[3]:>10.3f}")
    print(line)

    print(f"Imu pack stamp: {state.imu_stamp:<10}")
    print(
        f"Accelerometer (m/s^2): {state.imu_acc[0]:>17.3f} {state.imu_acc[1]:>17.3f} {state.imu_acc[2]:>17.3f}")
    print(
        f"Attitude      (Euler): {state.imu_euler[0]:>17.3f} {state.imu_euler[1]:>17.3f} {state.imu_euler[2]:>17.3f}")
    print(
        f"Gyroscope     (rad/s): {state.imu_gyro[0]:>17.3f} {state.imu_gyro[1]:>17.3f} {state.imu_gyro[2]:>17.3f}")
    print(line)

    print(
        f"Attitude(est) (Euler): {state.est_euler[0]:>17.3f} {state.est_euler[1]:>17.3f} {state.est_euler[2]:>17.3f}")
    print(
        f"COM Pos(est)      (m): {state.est_com_pos[0]:>17.3f} {state.est_com_pos[1]:>17.3f} {state.est_com_pos[2]:>17.3f}")
    print(
        f"COM Vel(est)    (m/s): {state.est_com_vel[0]:>17.3f} {state.est_com_vel[1]:>17.3f} {state.est_com_vel[2]:>17.3f}")
    print(line)

    print(
        f"Bus Information:      {state.bus_voltage:>18.3f} {state.bus_current:>17.3f} {state.bus_energy:>17.3f}")
    print(line)

    print(
        f"Remote Controller: {state.rc_du[0]:>21} {state.rc_du[1]:>10} {state.rc_du[2]:>13} {state.rc_du[3]:>10}")
    print(
        f"{state.rc_keys[0]:>40} {state.rc_keys[1]:>10} {state.rc_keys[2]:>13} {state.rc_keys[3]:>10}")
    print(line)