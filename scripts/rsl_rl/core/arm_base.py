import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
from core.Base import *
from grpc import insecure_channel
from core.config import Config
from protos import arm_service_pb2_grpc as arm_pb2_grpc
from protos import droid_msg_pb2 as msg_pb2
from enum import Enum

class ArmState(Enum):
    INIT = 0
    HELLOW = 1
    SHAKE  = 2
    BAOQUAN = 3
    HUISHOU = 4
    JIGUANG = 5
    TUOJU = 6
    NONE = -1


class ArmBase:
    def __init__(self, _cfg):
        self.cfg = _cfg
        self.armActions = self.cfg.env.num_arm_actions
        # grpc defines
        self.armConfigs = msg_pb2.DroidConfigs()
        self.armState = msg_pb2.DroidArmResponse()
        self.armCommand = msg_pb2.DroidCommandRequest()
        channel = insecure_channel(self.cfg.env.grpc_channel + ":50052")
        self.armStub = arm_pb2_grpc.ArmServiceStub(channel)
        init_command(self.armCommand, self.cfg.env.num_arm_actions)
        for idx in range(12):
            self.armCommand.finger.append(10)
        # 建立通信，获取机器人底层信息
        self.get_arm_config()
        self.get_arm_state()
        for idx in range(self.armActions):
            self.armCommand.position[idx] = self.armState.position[idx]
        # 电机空间控制或关节空间控制
        set_motor_mode(self.armCommand, self.armConfigs)

        # path
        self.dt = 0.01  # step时间步长
        self.td = 0.    # 轨迹结束时间
        self.tt = 0.    # 轨迹当前时间
        self.N: int = 0   # 动作的总行数
        self.Nt: int = 0  # 给定行数
        self.Nc: int = -1  # 当前行数
        self.St = ArmState.INIT  # 给定状态
        self.Sc = ArmState.NONE  # 当前状态
        self.qc = np.zeros(self.armActions, dtype=float)
        self.q0 = np.zeros(self.armActions, dtype=float)
        self.qd = np.zeros(self.armActions, dtype=float)
        self.qt = np.zeros(self.armActions, dtype=float)

        for idx in range(self.armActions):
            self.qc[idx] = self.armState.position[idx]
        self.qt[:] = self.qc[:]
        self.init_actions()

    def init_actions(self):
        self.init_action = np.array([[2.0, -0.00, 0.17, -0.00, 0.40, -0.00, 0.17, -0.00, 0.40]], dtype=float)

        self.hello_action = np.array([[2.0,   -0.52, 0.17, -0.00, 1.40,   1.1, 0.2, 0.0, 1.5],
                                      [1.0,   -0.52, 0.17, -0.00, 1.40,   1.1, 0.2, 0.5, 1.5],
                                      [1.0,   -0.52, 0.17, -0.00, 1.40,   1.1, 0.2,-0.5, 1.5],
                                      [1.0,   -0.52, 0.17, -0.00, 1.40,   1.1, 0.2, 0.5, 1.5],
                                      [1.0,   -0.52, 0.17, -0.00, 1.40,   1.1, 0.2, 0.0, 1.5]], dtype=float)

        self.huishou_action = np.array([[2.0,   -0.52, 0.17, -0.00, 1.40,   2.6, 0.45,-1.6, 0.0],
                                        [0.8,   -0.52, 0.17, -0.00, 1.40,   2.6, 0.25,-1.6, 0.5],
                                        [0.8,   -0.52, 0.17, -0.00, 1.40,   2.6, 0.65,-1.6, 0.0],
                                        [0.8,   -0.52, 0.17, -0.00, 1.40,   2.6, 0.25,-1.6, 0.5],
                                        [0.8,   -0.52, 0.17, -0.00, 1.40,   2.6, 0.65,-1.6, 0.0]], dtype=float)

        self.shake_action = np.array([[2.0,   -0.52, 0.17, -0.00, 1.40,   0.4, 0.2, 0, 1.3],
                                      [0.5,   -0.52, 0.17, -0.00, 1.40,   0.4, 0.2, 0, 1.2],
                                      [0.5,   -0.52, 0.17, -0.00, 1.40,   0.4, 0.2, 0, 1.4],
                                      [0.5,   -0.52, 0.17, -0.00, 1.40,   0.4, 0.2, 0, 1.2],
                                      [0.5,   -0.52, 0.17, -0.00, 1.40,   0.4, 0.2, 0, 1.4],
                                      [0.5,   -0.52, 0.17, -0.00, 1.40,   0.4, 0.2, 0, 1.2],
                                      [0.5,   -0.52, 0.17, -0.00, 1.40,   0.4, 0.2, 0, 1.4],
                                      [0.5,   -0.52, 0.17, -0.00, 1.40,   0.4, 0.2, 0, 1.3]], dtype=float)

        self.baoquan_action = np.array([[2.0,   0.77, 0.33, -0.73, 1.72,   0.67, 0.48, -0.70, 1.82],
                                        [0.5,   0.77, 0.33, -0.73, 1.62,   0.67, 0.48, -0.70, 1.72],
                                        [0.5,   0.77, 0.33, -0.73, 1.82,   0.67, 0.48, -0.70, 1.92],
                                        [0.5,   0.77, 0.33, -0.73, 1.62,   0.67, 0.48, -0.70, 1.72],
                                        [0.5,   0.77, 0.33, -0.73, 1.82,   0.67, 0.48, -0.70, 1.92],
                                        [0.5,   0.77, 0.33, -0.73, 1.72,   0.67, 0.48, -0.70, 1.82]], dtype=float)

        self.jigaung_action = np.array([[2.0,  1.5, -0.04, -1.35, 1.3,   1.7, 0.06, 0.13, 1.6],
                                        [0.5,  1.5, -0.04, -1.35, 1.3,   1.7, 0.06, 0.13, 1.3],
                                        [0.5,  1.5, -0.04, -1.35, 1.3,   1.7, 0.06, 0.13, 1.9],
                                        [0.5,  1.5, -0.04, -1.35, 1.3,   1.7, 0.06, 0.13, 1.3],
                                        [0.5,  1.5, -0.04, -1.35, 1.3,   1.7, 0.06, 0.13, 1.9],
                                        [0.5,  1.5, -0.04, -1.35, 1.3,   1.7, 0.06, 0.13, 1.6]], dtype=float)

        self.tuoju_action = np.array([[2.0,  0.9,  -0.05,  -0.17,   1.02,   0.9,  -0.05,  -0.17,   1.02]], dtype=float)

    def get_arm_config(self):
        empty_request = msg_pb2.Empty()
        self.armConfigs = self.armStub.GetArmConfig(empty_request)

    def get_arm_state(self):
        empty_request = msg_pb2.Empty()
        self.armState = self.armStub.GetArmState(empty_request)
        # print_state(self.armState, self.armConfigs)

    def set_arm_command(self):
        response = self.armStub.SetArmCommand(self.armCommand)
        if not response:  # Assuming the RPC method returns a response
            print("RPC failed")

    def set_arm_path(self, T, qd):
        s0, s1, st = 0.0, 0.0, 0.0
        tt = 0.0
        dt = 0.002
        q0 = [0.0] * self.armActions
        for idx in range(self.armActions):
            q0[idx] = self.armState.position[idx]
        timer = NanoSleep(2)  # 创建一个1毫秒的NanoSleep对象
        while tt < T + dt / 2.0:
            start_time = time.perf_counter()
            self.get_arm_state()
            st = min(tt / T, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0
            # print(self.armState.position, self.armState.velocity)
            for idx in range(self.armActions):  # 假设关节数量是18
                qt = s0 * q0[idx] + s1 * qd[idx]
                self.armCommand.position[idx] = qt
            self.set_arm_command()
            tt += dt
            timer.waiting(start_time)  # 等待下一个时间步长

    def testArm(self):
        T = 2  # 总时间
        D2R = math.pi / 180.0
        dt0 = [-30, 10, 0, 80, -30, 10, 0, 80]  # 假设 NMC 是一个定义好的常量，表示关节数量
        dt0 = [x * D2R for x in dt0]
        # 填充 dt1 和 dt2 列表
        dt1 = [-30 * D2R, 10 * D2R, 0, 100 * D2R, 30 * D2R, 10 * D2R, 0, 100 * D2R]
        dt2 = [30 * D2R, 10 * D2R, 0, 100 * D2R, -30 * D2R, 10 * D2R, 0, 100 * D2R]

        # 执行关节规划
        for i in range(2):
            print("wave round %d" % (i * 2 + 1))
            self.set_arm_path(T, dt1)
            print("wave round %d" % (i * 2 + 2))
            self.set_arm_path(T, dt2)
        print("return to zero")
        gBot.set_arm_path(T, dt0)

    def setTraj(self, td, qd):
        self.tt = 0.
        self.td = td
        self.q0[:] = self.qt[:]
        self.qd[:] = qd[:]

    def getTraj(self):
        st = min(self.tt / self.td, 1.0)
        s0 = 0.5 * (1.0 + math.cos(math.pi * st))
        s1 = 1 - s0
        self.qt = s0 * self.q0 + s1 * self.qd
        for idx in range(self.armActions):
            self.armCommand.position[idx] = self.qt[idx]

    def do_action(self, action):
        self.N = action.shape[0]
        if self.Nt <= self.N-1:    #动作结束判断
            if self.Nc != self.Nt: #行数更新，重新设置轨迹
                self.Nc = self.Nt
                self.tt = 0
                for i in range(self.armActions):
                    self.qt[i] = self.armCommand.position[i]
                self.setTraj(action[self.Nc][0], action[self.Nc][1:1+self.armActions])
            else:                  #计算轨迹
                self.tt += self.dt
                self.getTraj()
                if self.tt > action[self.Nc][0]: # 当前行轨迹执行完成，进入下一行
                    self.Nt += 1
        # print("St=%3d Sc=%3d Nt=%3d Nc=%3d tt=%7.3f qt=%7.3f" %(self.St.value, self.Sc.value, self.Nt, self.Nc, self.tt, self.qt[1]))

    def arm_task(self, state):
        if state.Y and not state.RB:
            self.St = ArmState.INIT
        elif state.X and not state.RB:
            self.St = ArmState.HELLOW
        elif state.A and  not state.RB:
            self.St = ArmState.HUISHOU
        elif state.Y and state.RB:
            self.St = ArmState.SHAKE
        elif state.X and state.RB:
            self.St = ArmState.JIGUANG

        if self.Sc != self.St: #状态更新，重新初始化动作的行数
            self.Sc = self.St
            self.tt = 0.
            self.Nt = 0
            self.Nc = -1

        if self.Sc == ArmState.INIT:
            self.do_action(self.init_action.copy())
        elif self.Sc == ArmState.HELLOW:
            self.do_action(self.hello_action.copy())
        elif self.Sc == ArmState.SHAKE:
            self.do_action(self.shake_action.copy())
        elif self.Sc == ArmState.BAOQUAN:
            self.do_action(self.baoquan_action.copy())
        elif self.Sc == ArmState.HUISHOU:
            self.do_action(self.huishou_action.copy())
        elif self.Sc == ArmState.JIGUANG:
            self.do_action(self.jigaung_action.copy())
        elif self.Sc == ArmState.TUOJU:
            self.do_action(self.tuoju_action.copy())

    def reset_arm_task(self):
        self.St = ArmState.INIT  # 给定状态
        self.Sc = ArmState.NONE  # 当前状态
        self.tt = 0.
        self.Nt = 0
        self.Nc = -1
        for idx in range(self.armActions):
            self.qc[idx] = self.armState.position[idx]
        self.qt[:] = self.qc[:]


if __name__ == '__main__':
    gBot = ArmBase(Config)
    gBot.testArm()


