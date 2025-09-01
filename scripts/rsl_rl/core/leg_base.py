import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
from core.Base import *
from grpc import insecure_channel
from core.config import Config
from protos  import leg_service_pb2_grpc as leg_pb2_grpc
from protos import droid_msg_pb2 as msg_pb2


class LegBase:
    def __init__(self, _cfg):
        self.cfg = _cfg
        self.legActions = self.cfg.env.num_leg_actions
        self.legConfigs = msg_pb2.DroidConfigs()
        self.legState = msg_pb2.DroidStateResponse()
        self.legCommand = msg_pb2.DroidCommandRequest()
        channel = insecure_channel(self.cfg.env.grpc_channel + ":50051")
        self.legStub = leg_pb2_grpc.LegServiceStub(channel)
        init_command(self.legCommand, self.cfg.env.num_leg_actions)
        # 建立通信，获取机器人底层信息
        self.get_leg_config()
        self.get_leg_state()
        # 电机空间控制或关节空间控制
        set_motor_mode(self.legCommand, self.legConfigs)
        # self.set_joint_mode(self.legCommand)

    def get_leg_config(self):
        empty_request = msg_pb2.Empty()
        self.legConfigs = self.legStub.GetLegConfig(empty_request)
        print_configs(self.legConfigs)

    def get_leg_state(self):
        empty_request = msg_pb2.Empty()
        self.legState = self.legStub.GetLegState(empty_request)
        # print_state(self.legState, self.legConfigs)

    def set_leg_command(self):
        response = self.legStub.SetLegCommand(self.legCommand)
        if not response:  # Assuming the RPC method returns a response
            print("RPC failed")

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

    def testLeg(self):
        T = 0.5  # 总时间
        dt0 = [0.] * self.legActions  # 假设 NMC 是一个定义好的常量，表示关节数量
        D2R = math.pi / 180.0
        # 填充 dt1 和 dt2 列表
        dt1 = [0, 0, 65 * D2R, -130 * D2R, 65 * D2R, 0, 0, 65 * D2R, -130 * D2R, 65 * D2R]
        dt2 = [0, 0, 5 * D2R, -10 * D2R, 5 * D2R, 0, 0, 5 * D2R, -10 * D2R, 5 * D2R]

        # 执行关节规划
        for i in range(2000):
            print("wave round %d" % (i * 2 + 1))
            self.set_leg_path(T, dt1)
            print("wave round %d" % (i * 2 + 2))
            self.set_leg_path(T, dt2)
        print("return to zero")
        gBot.set_leg_path(T, dt0)


if __name__ == '__main__':
    gBot = LegBase(Config)
    gBot.testLeg()
