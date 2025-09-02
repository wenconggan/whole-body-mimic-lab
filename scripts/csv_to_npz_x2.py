"""
该脚本的核心功能：将 CSV 格式的机器人运动数据重放（在 Isaac Sim 中可视化），并转换为 NPZ 格式保存。
NPZ 格式更适合后续强化学习训练（如运动跟踪任务）中加载，支持快速读取关节位置、速度、躯干姿态等关键数据。

使用示例（命令行）：
# 基础用法：指定输入 CSV、帧率、输出 NPZ 路径
python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
--output_file ./motions/dance1_subject2.npz --output_fps 50

# 实际项目用法（针对 G1 机器人）：
python scripts/csv_to_npz.py --input_file /home/jojo/PycharmProjects/whole_body_tracking/retargeted_motions/g1/dance1_subject1.csv --input_fps 30 --frame_range 122 722 \
--output_file /home/jojo/PycharmProjects/whole_body_tracking/retargeted_motions/g1/dance1_subject1.npz --output_fps 50
"""

"""
注意：运行脚本前必须先启动 Isaac Sim 仿真器，否则会报错。
Isaac Sim 是 NVIDIA 用于机器人仿真的核心工具，负责物理渲染、关节状态计算等。
"""

# 导入基础工具库
import argparse  # 解析命令行参数
import numpy as np  # 处理数值计算（如数组、矩阵）
import os  # 处理文件路径、系统环境变量

# 导入 Isaac Lab 相关模块（机器人仿真核心依赖）
from isaaclab.app import AppLauncher  # 启动 Isaac Sim 应用的工具

# 1. 解析命令行参数：定义脚本可接收的输入参数
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")

# 核心输入参数：CSV 运动文件路径（必须指定）
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
# 输入 CSV 的帧率（默认 30 FPS，需与 CSV 数据采集时的帧率一致）
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
# 帧范围：截取 CSV 中的部分帧（START 和 END 均为包含性索引，从 1 开始；不指定则加载全部帧）
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
# 输出 NPZ 的名称（用于后续标识运动数据，如 "dance1_subject1"）
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
# 输出 NPZ 的帧率（默认 50 FPS，可根据训练需求调整，如 Isaac Lab 常用 50 FPS）
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
# 新增参数 1：禁用 WandB 日志上传（WandB 是实验管理工具，禁用后仅本地保存）
parser.add_argument("--no_wandb", action="store_true", default=True,help="Disable WandB logging and registry upload.")
# 新增参数 2：NPZ 文件的本地保存路径（默认 /tmp/motion.npz，建议指定项目内路径）
parser.add_argument("--save_to", type=str, default="/home/wenconggan/whole_body_tracking/motion/", help="Path to save the generated npz.")

# 附加 Isaac Sim 启动参数（如 --headless 无头模式、--device GPU 设备）
AppLauncher.add_app_launcher_args(parser)
# 解析所有命令行参数，存储到 args_cli 对象中
args_cli = parser.parse_args()

# 2. 启动 Isaac Sim 仿真应用
app_launcher = AppLauncher(args_cli)  # 初始化启动器
simulation_app = app_launcher.app  # 获取仿真应用实例（核心，后续所有仿真操作依赖此对象）

"""
以上是仿真环境初始化，以下是运动数据处理和重放的核心逻辑
"""

# 导入 PyTorch（Isaac Lab 常用张量计算库）
import torch

# 导入 Isaac Lab 仿真、资产、场景相关工具
import isaaclab.sim as sim_utils  # 仿真工具（如相机控制、物理步长）
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # 资产配置（关节型机器人、基础资产）
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # 交互式场景（管理机器人、地面、灯光）
from isaaclab.sim import SimulationContext  # 仿真上下文（控制仿真循环、设备管理）
from isaaclab.utils import configclass  # 配置类装饰器（确保配置参数结构化）
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # Isaac Lab 内置资源目录（如天空盒、材质）
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp  # 数学工具（四元数、轴角转换）

##
# 预定义配置：导入 G1 机器人的圆柱碰撞体配置（G1 是四足/人形机器人模型）
##
from whole_body_tracking.robots.x2 import x2_CYLINDER_CFG


# 3. 定义场景配置类：描述仿真场景的组成（地面、灯光、机器人）
@configclass  # Isaac Lab 装饰器，使类支持配置参数的结构化和校验
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # 地面平面：使用默认地面，无特殊配置
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 天空灯光：使用 HDR 天空盒，提供真实光照效果
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",  # 灯光在 USD 场景中的路径
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,  # 光照强度（数值越大越亮）
            # 天空盒纹理路径：使用 Isaac Lab 内置的 HDR 纹理（模拟真实天空光照）
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # 关节型资产（机器人）：基于 G1 机器人的圆柱碰撞体配置，修改路径以支持多环境
    robot: ArticulationCfg = x2_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # {ENV_REGEX_NS} 是命名空间占位符，此处仅单环境（num_envs=1），实际会替换为 /World/Env_0


# 4. 定义运动加载器类：负责读取 CSV 数据、插值到目标帧率、计算速度
class MotionLoader:
    def __init__(
        self,
        motion_file: str,  # CSV 运动文件路径
        input_fps: int,    # 输入 CSV 的帧率
        output_fps: int,   # 输出 NPZ 的帧率
        device: torch.device,  # 计算设备（CPU/GPU，Isaac Lab 优先用 GPU）
        frame_range: tuple[int, int] | None,  # 截取的帧范围（None 表示全部）
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps  # 输入帧的时间间隔（秒）
        self.output_dt = 1.0 / self.output_fps  # 输出帧的时间间隔（秒）
        self.current_idx = 0  # 当前重放的帧索引（用于循环取帧）
        self.device = device
        self.frame_range = frame_range

        # 初始化时执行核心流程：加载 CSV → 插值到目标帧率 → 计算速度
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """从 CSV 文件加载运动数据，提取基础姿态（位置、旋转）和关节位置"""
        # 读取 CSV 文件：根据帧范围决定是否跳过行/限制最大行数
        if self.frame_range is None:
            # 加载全部帧：np.loadtxt 读取 CSV，逗号分隔
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            # 加载指定帧范围：skiprows 跳过前 N 行，max_rows 限制读取行数
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,  # 从 START 帧开始（索引从 1 转 0）
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,  # 读取 END-START+1 帧
                )
            )
        # 转换数据类型和设备：float32 节省内存，转移到指定设备（GPU/CPU）
        motion = motion.to(torch.float32).to(self.device)

        # 解析 CSV 数据列（CSV 格式约定：前 3 列是基础位置，4-7 是基础旋转，7+ 是关节位置）
        self.motion_base_poss_input = motion[:, :3]  # 基础位置（x, y, z）
        self.motion_base_rots_input = motion[:, 3:7]  # 基础旋转（原 CSV 可能是 xyzw，此处转为 wxyz）
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # 列重排：xyzw → wxyz（Isaac Lab 标准）
        self.motion_dof_poss_input = motion[:, 7:]  # 关节位置（DOF：Degree of Freedom，自由度）

        # 计算运动的基础信息
        self.input_frames = motion.shape[0]  # 输入帧总数
        self.duration = (self.input_frames - 1) * self.input_dt  # 运动总时长（秒）
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """将输入运动插值到目标帧率（如 30 FPS → 50 FPS），避免帧率不匹配导致的卡顿"""
        # 生成输出帧率对应的时间序列（从 0 到总时长，步长为输出帧间隔）
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]  # 输出帧总数

        # 计算每个输出时间点对应的输入帧索引和混合系数（用于插值）
        index_0, index_1, blend = self._compute_frame_blend(times)

        # 线性插值（LERP）：基础位置、关节位置（适合平移/线性量）
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],  # 前一帧位置
            self.motion_base_poss_input[index_1],  # 后一帧位置
            blend.unsqueeze(1),  # 混合系数（扩展为列向量，匹配位置的 3 列维度）
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

        # 球面线性插值（SLERP）：基础旋转（四元数，适合旋转量，避免插值后的旋转失真）
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )

        # 打印插值结果
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """线性插值（Linear Interpolation）：计算 a 和 b 之间的混合值，blend ∈ [0,1]"""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """球面线性插值（Spherical Linear Interpolation）：用于四元数旋转插值，保持旋转精度"""
        slerped_quats = torch.zeros_like(a)  # 初始化输出张量（与输入形状相同）
        for i in range(a.shape[0]):
            # 逐帧插值：quat_slerp 是 Isaac Lab 封装的四元数插值函数
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """计算每个输出时间点对应的输入帧索引和混合系数"""
        phase = times / self.duration  # 时间相位（0 → 1，表示运动的进度）
        # 前一帧索引：floor 取整（如 phase=0.3 → 0.3*(N-1) → 向下取整）
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        # 后一帧索引：前一帧 +1，且不超过最后一帧（避免越界）
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        # 混合系数：phase*(N-1) - index_0（表示在 index_0 和 index_1 之间的比例）
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """计算运动的速度（线性速度、角速度、关节速度），用于后续训练中的状态输入"""
        # 基础线性速度：对位置求梯度（数值微分），步长为输出帧间隔
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        # 关节速度：对关节位置求梯度
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        # 基础角速度：通过四元数的导数计算（SO3 空间的微分，确保旋转速度的正确性）
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """计算 SO3 旋转序列的导数（角速度），输入为四元数序列，输出为轴角形式的角速度"""
        # 取前 n-2 帧和后 n-2 帧（中心差分，提高精度）
        q_prev, q_next = rotations[:-2], rotations[2:]
        # 计算相邻两帧的相对旋转：q_next * q_prev的共轭（表示从 q_prev 到 q_next 的旋转）
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        # 将相对旋转四元数转为轴角（axis-angle），再除以 2*dt（中心差分的步长）
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        # 补充首尾帧的角速度（重复第一帧和最后一帧，确保输出长度与输入一致）
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(
            self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """获取运动的下一帧状态，包括位置、旋转和速度信息"""
        # 提取当前帧的运动状态（使用切片保持维度为[1, N]，适配单环境批量处理）
        state = (
            self.motion_base_poss[self.current_idx: self.current_idx + 1],  # 机器人基座位置 (x, y, z)
            self.motion_base_rots[self.current_idx: self.current_idx + 1],  # 机器人基座旋转（四元数 wxyz）
            self.motion_base_lin_vels[self.current_idx: self.current_idx + 1],  # 基座线速度 (vx, vy, vz)
            self.motion_base_ang_vels[self.current_idx: self.current_idx + 1],  # 基座角速度 (wx, wy, wz)
            self.motion_dof_poss[self.current_idx: self.current_idx + 1],  # 关节位置（所有自由度）
            self.motion_dof_vels[self.current_idx: self.current_idx + 1],  # 关节速度（所有自由度）
        )
        # 帧索引递增（准备下一帧）
        self.current_idx += 1
        reset_flag = False  # 重置标志：标记是否循环到运动起点
        # 若当前索引超过总帧数，重置索引并触发循环标志
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag  # 返回当前状态和重置标志

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """运行仿真循环：加载运动数据，在Isaac Sim中重放，并记录数据到NPZ文件"""
    # 初始化运动加载器：读取CSV并预处理（插值、计算速度）
    motion = MotionLoader(
        motion_file=args_cli.input_file,  # 输入CSV路径
        input_fps=args_cli.input_fps,  # 输入帧率
        output_fps=args_cli.output_fps,  # 输出帧率
        device=sim.device,  # 计算设备（GPU/CPU）
        frame_range=args_cli.frame_range,  # 截取的帧范围（可选）
    )

    # 提取场景中的机器人资产和关节索引
    robot = scene["robot"]  # 从场景中获取机器人实例
    # 查找关节名称对应的索引（确保与CSV中关节顺序匹配）
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- 数据日志器：记录重放过程中的机器人状态 ---------------------------
    log = {
        "fps": [args_cli.output_fps],  # 输出帧率（元组形式存为数组）
        "joint_pos": [],  # 关节位置序列 [帧数量, 关节数量]
        "joint_vel": [],  # 关节速度序列 [帧数量, 关节数量]
        "body_pos_w": [],  # 身体部位位置（世界坐标系）[帧数量, 部位数量, 3]
        "body_quat_w": [],  # 身体部位旋转（世界坐标系，四元数）[帧数量, 部位数量, 4]
        "body_lin_vel_w": [],  # 身体部位线速度（世界坐标系）[帧数量, 部位数量, 3]
        "body_ang_vel_w": [],  # 身体部位角速度（世界坐标系）[帧数量, 部位数量, 3]
    }
    file_saved = False  # 标记文件是否已保存（避免重复保存）
    # --------------------------------------------------------------------------

    # 仿真循环：持续运行直到Isaac Sim窗口关闭
    while simulation_app.is_running():
        # 获取下一帧运动状态和重置标志
        (
            (
                motion_base_pos,  # 基座位置
                motion_base_rot,  # 基座旋转
                motion_base_lin_vel,  # 基座线速度
                motion_base_ang_vel,  # 基座角速度
                motion_dof_pos,  # 关节位置
                motion_dof_vel,  # 关节速度
            ),
            reset_flag,  # 运动是否循环到起点
        ) = motion.get_next_state()

        # ---------------------- 设置机器人基座状态 ----------------------
        # 克隆默认基座状态（包含初始位姿、速度等）
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos  # 覆盖位置
        # 将基座位置偏移到环境原点（多环境时避免重叠，此处单环境仍保留逻辑）
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot  # 覆盖旋转
        root_states[:, 7:10] = motion_base_lin_vel  # 覆盖线速度
        root_states[:, 10:] = motion_base_ang_vel  # 覆盖角速度
        # 将基座状态写入仿真器（更新物理引擎中的机器人位置）
        robot.write_root_state_to_sim(root_states)

        # ---------------------- 设置机器人关节状态 ----------------------
        # 克隆默认关节状态（包含初始位置、速度）
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        # 用运动数据覆盖关节位置和速度（通过索引匹配对应关节）
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        # 将关节状态写入仿真器（更新物理引擎中的关节角度）
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # 渲染画面（仅更新视觉，不执行物理模拟步骤，因为是重放预定义运动）
        sim.render()
        # 更新场景状态（基于当前物理时间步长）
        scene.update(sim.get_physics_dt())

        # 调整相机视角：始终跟踪机器人基座（偏移量 [2, 2, 0.5] 确保完整视角）
        pos_lookat = root_states[0, :3].cpu().numpy()  # 基座位置（转为numpy数组）
        sim.set_camera_view(
            eye=pos_lookat + np.array([2.0, 2.0, 0.5]),  # 相机位置
            target=pos_lookat  # 相机朝向目标（机器人基座）
        )

        # ---------------------- 记录运动数据（未保存时） ----------------------
        if not file_saved:
            # 记录关节状态（从机器人数据中读取，确保与仿真器状态一致）
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            # 记录身体部位状态（世界坐标系下的位置、旋转、速度）
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        # ---------------------- 运动循环结束时保存NPZ文件 ----------------------
        if reset_flag and not file_saved:
            file_saved = True  # 标记为已保存，避免重复执行
            # 将列表转换为numpy数组（形状：[帧数量, ...]）
            for k in (
                    "joint_pos", "joint_vel",
                    "body_pos_w", "body_quat_w",
                    "body_lin_vel_w", "body_ang_vel_w"
            ):
                log[k] = np.stack(log[k], axis=0)

            # 本地保存NPZ文件（必选）
            np.savez(args_cli.save_to, **log)
            print(f"[INFO]: 运动数据已保存至: {args_cli.save_to}")

            # 可选：上传到WandB（若未禁用且环境变量允许）
            use_wandb = (not args_cli.no_wandb) and (
                        os.environ.get("WANDB_DISABLED", "").lower() not in ["1", "true", "yes"])
            if use_wandb:
                import wandb  # 延迟导入，避免未安装时出错

                COLLECTION = args_cli.output_name  # 运动数据名称（用于WandB标识）
                run = wandb.init(project="csv_to_npz", name=COLLECTION)  # 初始化WandB运行
                print(f"[INFO]: 正在将运动数据上传至WandB: {COLLECTION}")
                REGISTRY = "motions"  # 注册到WandB的"motions"集合
                # 记录NPZ文件为WandB资产
                logged_artifact = run.log_artifact(
                    artifact_or_path=args_cli.save_to,
                    name=COLLECTION,
                    type=REGISTRY
                )
                try:
                    # 链接资产到公共注册表（方便后续复用）
                    run.link_artifact(
                        artifact=logged_artifact,
                        target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}"
                    )
                    print(f"[INFO]: 运动数据已保存至WandB注册表: {REGISTRY}/{COLLECTION}")
                except Exception as e:
                    print(f"[WARN]: 跳过注册表链接: {e}")

def main():
    """主函数：初始化仿真环境并启动运动重放"""
    # 配置仿真参数
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)  # 指定计算设备（如GPU）
    sim_cfg.dt = 1.0 / args_cli.output_fps  # 仿真时间步长（与输出帧率匹配）
    sim = SimulationContext(sim_cfg)  # 创建仿真上下文

    # 设计场景：包含地面、灯光和机器人
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)  # 单环境，间距2.0
    scene = InteractiveScene(scene_cfg)  # 创建交互式场景

    # 重置仿真环境（加载资产，初始化物理状态）
    sim.reset()
    print("[INFO]: 环境设置完成...")

    # 启动仿真循环（传入关节名称列表，确保与CSV数据匹配）
    run_simulator(
        sim,
        scene,
        joint_names=[  # G1机器人的关节名称列表（需与CSV中关节顺序一致）
            "L_hip_yaw",  # 左髋偏航关节
            "L_hip_roll",  # 左髋滚动关节
            "L_hip_pitch",  # 左髋俯仰关节
            "L_knee_pitch",  # 左膝关节
            "L_ankle_pitch",  # 左踝俯仰关节
            "R_hip_yaw",  # 右髋俯仰关节
            "R_hip_roll",  # 右髋滚动关节
            "R_hip_pitch",  # 右髋偏航关节
            "R_knee_pitch",  # 右膝关节
            "R_ankle_pitch",  # 右踝俯仰关
            "L_shoulder_pitch",  # 左肩俯仰关节
            "L_shoulder_roll",  # 左肩滚动关节
            "L_shoulder_yaw",  # 左肩偏航关节
            "L_elbow_pitch",  # 左肘关节
            "R_shoulder_pitch",  # 右肩俯仰关节
            "R_shoulder_roll",  # 右肩滚动关节
            "R_shoulder_yaw",  # 右肩偏航关节
            "R_elbow_pitch",  # 右肘关节
        ],
    )

if __name__ == "__main__":
    # 执行主函数
    main()
    # 关闭仿真应用（释放资源）
    simulation_app.close()

