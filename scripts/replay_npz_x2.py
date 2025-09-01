"""
该脚本演示如何使用Isaac Lab的交互式场景接口，搭建包含x2机器人的场景，并实现运动文件（NPZ格式）的重放功能。
核心逻辑：加载预转换的运动数据（关节角度、根节点位姿），在仿真中驱动机器人复现运动，支持可视化观察。

使用方式（在终端执行）：
.. code-block:: bash
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/x2/motions/lafan_walk_short.npz
"""

"""第一步：先启动Isaac Sim仿真器（必须在导入其他仿真模块前执行）"""

import argparse  # 用于解析命令行参数
import numpy as np  # 用于数值计算（如相机视角调整）
import torch  # 用于张量运算（Isaac Lab核心数据类型）

from isaaclab.app import AppLauncher  # Isaac Lab的应用启动器，负责初始化Isaac Sim

# 1. 配置命令行参数（用户可通过终端传入参数，如运动文件路径、WandB注册表名）
parser = argparse.ArgumentParser(description="Replay converted motions.")  # 创建参数解析器
# --registry_name：WandB注册表名（用于从云端加载运动文件，非必填）
parser.add_argument("--registry_name", type=str, required=False, help="The name of the wand registry.")
# --motion_file：本地运动NPZ文件路径（优先使用，覆盖云端加载，非必填但推荐）
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion npz (overrides registry)")

# 2. 追加Isaac Lab应用启动器的默认参数（如仿真设备、无头模式等）
AppLauncher.add_app_launcher_args(parser)
# 3. 解析命令行传入的参数（存储到args_cli中，后续使用）
args_cli = parser.parse_args()

# 4. 启动Isaac Sim仿真应用（根据命令行参数初始化，如--headless启用无头模式）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # 获取仿真应用实例，用于后续判断仿真是否运行

"""第二步：导入仿真核心模块（必须在启动仿真器后导入）"""

import isaaclab.sim as sim_utils  # Isaac Lab仿真工具库（场景设置、物理步长控制等）
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # 资产类（关节机器人、基础资产）
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # 交互式场景类（管理所有实体）
from isaaclab.sim import SimulationContext  # 仿真上下文（管理物理引擎、设备等）
from isaaclab.utils import configclass  # 配置类装饰器（用于定义结构化场景配置）
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # Isaac Lab核心资产路径（如天空盒纹理）

## 导入自定义模块（用户根据自己的项目路径调整）
from whole_body_tracking.robots.x2 import x2_CYLINDER_CFG  # x2机器人的预定义配置（关节、URDF路径等）
from whole_body_tracking.tasks.tracking.mdp import MotionLoader  # 运动加载器（解析NPZ运动文件）


# 1. 定义场景配置类（使用@configclass装饰器，确保参数结构化）
@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """运动重放场景的配置类，继承自基础交互式场景配置，定义场景中的所有实体"""

    # 1.1 地面资产配置（基础平面，用于机器人站立）
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",  # 地面在USD场景中的路径（唯一标识）
        spawn=sim_utils.GroundPlaneCfg()  # 地面生成配置（默认参数：尺寸100x100m，灰色材质）
    )

    # 1.2 天空灯资产配置（提供场景光照，影响可视化效果）
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",  # 天空灯在USD场景中的路径
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,  # 光照强度（数值越大场景越亮）
            # 天空盒纹理路径（使用Isaac Lab内置的HDR纹理，模拟真实天空光照）
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # 1.3 x2机器人资产配置（关节型机器人，核心运动实体）
    # 从预定义的x2_CYLINDER_CFG复制配置，并修改prim_path适配多环境（{ENV_REGEX_NS}是环境命名空间占位符）
    robot: ArticulationCfg = x2_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# 2. 仿真运行函数（核心逻辑：读取运动数据，驱动机器人运动，更新仿真状态）
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """
    仿真主循环函数，负责运动数据加载和机器人状态更新。

    Args:
        sim: 仿真上下文实例（管理物理步长、渲染、相机等）
        scene: 交互式场景实例（管理机器人、地面等所有实体）
    """
    # 2.1 从场景中提取x2机器人实体（根据配置中的"robot"名称获取）
    robot: Articulation = scene["robot"]
    # 2.2 获取仿真物理步长（单位：秒，控制每次仿真更新的时间间隔）
    sim_dt = sim.get_physics_dt()

    # 2.3 确定运动文件来源（优先使用本地文件，无本地文件则从WandB云端加载）
    if args_cli.motion_file is not None:
        # 情况1：用户传入本地运动文件路径（如--motion_file xxx.npz）
        motion_file = args_cli.motion_file
    else:
        # 情况2：从WandB注册表加载（需用户提供--registry_name）
        if args_cli.registry_name is None:
            raise ValueError("Please provide either --motion_file or --registry_name")
        registry_name = args_cli.registry_name
        # 若注册表名不含版本号（如":latest"），自动追加最新版本
        if ":" not in registry_name:
            registry_name += ":latest"
        # 导入WandB模块（用于云端资产下载）
        import pathlib
        import wandb
        # 初始化WandB API，下载运动文件到本地
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        # 获取下载后的本地运动文件路径（默认下载到./artifacts/目录）
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # 2.4 加载运动数据（使用自定义的MotionLoader，解析NPZ文件中的关节角度、根节点位姿等）
    motion = MotionLoader(
        motion_file,  # 运动文件路径
        torch.tensor([0], dtype=torch.long, device=sim.device),  # 环境索引（单环境为[0]）
        sim.device,  # 数据存储设备（CPU/GPU，与仿真设备一致）
    )
    # 初始化时间步计数器（记录当前重放的运动帧，单环境为1个元素的张量）
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # 2.5 仿真主循环（持续运行直到用户关闭Isaac Sim）
    while simulation_app.is_running():
        # 2.5.1 更新时间步（每帧+1，达到运动总帧数后重置为0，实现循环重放）
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total  # 标记需要重置的环境（单环境为[True/False]）
        time_steps[reset_ids] = 0  # 重置时间步为0，重新开始重放

        # 2.5.2 构建机器人根节点状态（位置、旋转、线速度、角速度）
        root_states = robot.data.default_root_state.clone()  # 复制默认根节点状态（作为基础）
        # 根节点位置：从运动数据中读取当前帧位置，并加上环境原点（适配多环境偏移）
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        # 根节点旋转：从运动数据中读取当前帧四元数（w,x,y,z格式）
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        # 根节点线速度：从运动数据中读取当前帧世界坐标系下的线速度
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        # 根节点角速度：从运动数据中读取当前帧世界坐标系下的角速度
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        # 2.5.3 将机器人状态写入仿真器（更新物理引擎中的机器人状态）
        robot.write_root_state_to_sim(root_states)  # 写入根节点状态
        # 写入关节状态（位置和速度，从运动数据中读取当前帧）
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        # 将场景中所有实体的状态写入仿真器（如地面、天空灯，此处无变化但需调用）
        scene.write_data_to_sim()

        # 2.5.4 渲染场景（仅更新可视化，不执行物理步长sim.step()，因为运动重放无需物理计算）
        sim.render()
        # 更新场景中所有实体的内部数据缓冲区（与仿真器状态同步）
        scene.update(sim_dt)

        # 2.5.5 调整相机视角（跟随机器人根节点，方便观察运动）
        # 获取机器人根节点当前位置（单环境，取第0个环境的位置，转换为numpy数组）
        pos_lookat = root_states[0, :3].cpu().numpy()
        # 设置相机位置：在机器人前方2m、右侧2m、上方0.5m处，看向机器人根节点
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


# 3. 主函数（初始化仿真和场景，启动仿真循环）
def main():
    """主函数：初始化仿真上下文和场景，启动运动重放"""
    # 3.1 配置仿真参数（设备、步长等）
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)  # 仿真设备（从命令行参数获取，如--device cuda:0）
    sim_cfg.dt = 0.02  # 仿真物理步长（0.02秒/帧，对应50Hz，与常见运动数据帧率匹配）
    # 创建仿真上下文实例（初始化物理引擎、USD场景等）
    sim = SimulationContext(sim_cfg)

    # 3.2 配置场景参数（环境数量、间距等）
    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=1,  # 仿真环境数量（单环境，运动重放无需多环境）
        env_spacing=2.0  # 多环境时间距（单环境时无效，仅为占位）
    )
    # 创建交互式场景实例（根据配置加载地面、天空灯、机器人）
    scene = InteractiveScene(scene_cfg)
    # 重置仿真器（初始化所有实体状态，确保启动时状态正确）
    sim.reset()

    # 3.3 启动仿真循环（传入仿真上下文和场景，开始运动重放）
    run_simulator(sim, scene)


# 4. 脚本入口（当脚本被直接运行时执行）
if __name__ == "__main__":
    # 运行主函数（初始化并启动仿真）
    main()
    # 关闭仿真应用（退出时释放资源，避免内存泄漏）
    simulation_app.close()