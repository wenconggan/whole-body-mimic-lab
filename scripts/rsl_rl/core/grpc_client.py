import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
import math
import time
import threading
from base64 import b64encode
import asyncio
from grpc import insecure_channel
from core.arm_base import ArmBase
from core.leg_base import LegBase


class NanoSleep:
    def __init__(self, ms):
        self.duration_sec = ms * 0.001  # 转化为单位秒

    def waiting(self, _start_time):
        while True:
            current_time = time.perf_counter()
            elapsed_time = current_time - _start_time
            if elapsed_time >= self.duration_sec:
                break

class DroidGrpcClient(ArmBase, LegBase):
    def __init__(self, _cfg):
        ArmBase.__init__(self, _cfg)
        LegBase.__init__(self, _cfg)
        self.robot_actions = self.armActions + self.legActions

        self.robot_actions =  self.legActions

    def get_robot_config(self):
        self.get_arm_config()
        self.get_leg_config()

    def get_robot_state(self):
        self.get_arm_state()
        self.get_leg_state()

    def set_robot_command(self):
        self.set_arm_command()
        self.set_leg_command()
    def set_leg_path(self, T, qd):
        s0, s1, st = 0.0, 0.0, 0.0
        tt = 0.0
        dt = 0.002
        q0 = [0.0] * self.legActions
        for idx in range(self.legActions):
            q0[idx] = self.legState.position[idx]
        timer = NanoSleep(2)  # 创建一个1毫秒的NanoSleep对象
        while tt < T + dt / 2.0:
            start_time = time.perf_counter()
            self.get_leg_state()
            st = min(tt / T, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0

            for idx in range(self.legActions):  # 假设关节数量是18
                qt = s0 * q0[idx] + s1 * qd[idx]
                self.legCommand.position[idx] = qt
            self.set_leg_command()
            tt += dt
            timer.waiting(start_time)  # 等待下一个时间步长

    def joint_plan(self, T, qd):
        s0, s1, st = 0.0, 0.0, 0.0
        tt = 0.0
        dt = 0.002
        q0 = [0.0] * self.legActions
        for idx in range(self.legActions):
            q0[idx] = self.legState.position[idx]
        timer = NanoSleep(2)  # 创建一个1毫秒的NanoSleep对象
        while tt < T + dt / 2.0:
            start_time = time.perf_counter()
            self.get_robot_state()
            st = min(tt / T, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0
            for idx in range(self.legActions):  # 假设关节数量是18
                qt = s0 * q0[idx] + s1 * qd[idx]
                self.legCommand.position[idx] = qt
            self.set_robot_command()
            tt += dt
            timer.waiting(start_time)  # 等待下一个时间步长

    def print_robot_configs(self):
        line = '-' * 200
        print("---------+" + line)
        print("MtrName  |", end="")
        for i in range(len(self.robotConfigs.joint_name)):
            print(f"{self.robotConfigs.joint_name[i]:>11}", end="")
        print()
        print("---------+" + line)
        for attr in ["ecatid", "channel", "deviceid", "aeofst", "uie", "uic", "uae", "pmin", "pmax", "ae0", "kp", "kd",
                     "imax"]:
            print("{:<8} |".format(attr.upper()), end="")
            for i in range(self.legActions):
                print("{:>11.3f}".format(getattr(self.robotConfigs, attr)[i]), end="")
            print()
        print("---------+" + line)

    def print_robot_state(self):
        line = '-' * 200
        print(f"system tic: {self.legState.system_tic} ms")
        print("---------+" + line)
        print("MtrName  |", end="")
        for i in range(len(self.robotConfigs.joint_name)):
            print(f"{self.robotConfigs.joint_name[i]:>11}", end="")
        print()
        print("---------+" + line)

        labels = ["qc", "dqc", "tqc", "temp", "absc", "loss"]
        for label in labels:
            print(f"{label:<8} |", end="")
            for i in range(len(self.robotConfigs.joint_name)):
                if label == "qc":
                    print(f"{self.legState.position[i]:>11.3f}", end="")
                elif label == "dqc":
                    print(f"{self.legState.velocity[i]:>11.3f}", end="")
                elif label == "tqc":
                    print(f"{self.legState.torque[i]:>11.3f}", end="")
                elif label == "temp":
                    print(f"{self.legState.temperature[i]:>11.3f}", end="")
                elif label == "absc":
                    print(f"{self.legState.abs_encoder[i]:>11.3f}", end="")
                elif label == "loss":
                    print(f"{self.legState.pack_loss[i]:>11}", end="")
            print()
        print("---------+" + line)
        line = '-' * 80
        print(
            f"Foot Sensor (L L R R): {self.legState.foot_force[0]:>10.3f} {self.legState.foot_force[1]:>10.3f} {self.legState.foot_force[2]:>15.3f} {self.legState.foot_force[3]:>10.3f}")
        print(line)

        print(f"Imu pack stamp: {self.legState.imu_stamp:<10}")
        print(
            f"Accelerometer (m/s^2): {self.legState.imu_acc[0]:>17.3f} {self.legState.imu_acc[1]:>17.3f} {self.legState.imu_acc[2]:>17.3f}")
        print(
            f"Attitude      (Euler): {self.legState.imu_euler[0]:>17.3f} {self.legState.imu_euler[1]:>17.3f} {self.legState.imu_euler[2]:>17.3f}")
        print(
            f"Gyroscope     (rad/s): {self.legState.imu_gyro[0]:>17.3f} {self.legState.imu_gyro[1]:>17.3f} {self.legState.imu_gyro[2]:>17.3f}")
        print(line)

        print(
            f"Attitude(est) (Euler): {self.legState.est_euler[0]:>17.3f} {self.legState.est_euler[1]:>17.3f} {self.legState.est_euler[2]:>17.3f}")
        print(
            f"COM Pos(est)      (m): {self.legState.est_com_pos[0]:>17.3f} {self.legState.est_com_pos[1]:>17.3f} {self.legState.est_com_pos[2]:>17.3f}")
        print(
            f"COM Vel(est)    (m/s): {self.legState.est_com_vel[0]:>17.3f} {self.legState.est_com_vel[1]:>17.3f} {self.legState.est_com_vel[2]:>17.3f}")
        print(line)

        print(
            f"Bus Information:      {self.legState.bus_voltage:>18.3f} {self.legState.bus_current:>17.3f} {self.legState.bus_energy:>17.3f}")
        print(line)

        print(
            f"Remote Controller: {self.legState.rc_du[0]:>21} {self.legState.rc_du[1]:>10} {self.legState.rc_du[2]:>13} {self.legState.rc_du[3]:>10}")
        print(
            f"{self.legState.rc_keys[0]:>40} {self.legState.rc_keys[1]:>10} {self.legState.rc_keys[2]:>13} {self.legState.rc_keys[3]:>10}")
        print(line)


if __name__ == '__main__':
    channel = insecure_channel('192.168.55.13:50051')
    gBot = DroidGrpcClient(channel)
    time.sleep(2)
    gBot.get_robot_config()
    gBot.get_robot_state()

    T = 0.6  # 总时间
    dt0 = [0.] * 18  # 假设 NMC 是一个定义好的常量，表示关节数量
    dt1 = [0.] * 18  # 创建一个 NMC 长度的列表，初始值为 0
    dt2 = [0.] * 18
    D2R = math.pi / 180.0

    dt1[1] = 10 * D2R
    dt1[2] = 60 * D2R
    dt1[3] = -80 * D2R
    dt1[4] = -80 * D2R
    dt1[10] = 0 * D2R
    dt1[13] = 100 * D2R
    dt1[14] = 0.0 * D2R
    dt1[17] = 100 * D2R

    dt2[6] = 10 * D2R
    dt2[7] = 60 * D2R
    dt2[8] = -80 * D2R
    dt2[9] = -80 * D2R
    dt2[10] = 0 * D2R
    dt2[13] = 100 * D2R
    dt2[14] = 0.0 * D2R
    dt2[17] = 100 * D2R

    for i in range(10000):
        print("wave round %d" % (i * 2 + 1))
        dt1[10] = gBot.legState.position[2]-0.5
        dt2[10] = gBot.legState.position[7]-0.5
        dt1[14] = gBot.legState.position[7]-0.5
        dt2[14] = gBot.legState.position[2]-0.5
        gBot.joint_plan(T, dt1)
        print("wave round %d" % (i * 2 + 2))
        gBot.joint_plan(T, dt2)
    print("return to zero")
    gBot.joint_plan(T, dt0)
