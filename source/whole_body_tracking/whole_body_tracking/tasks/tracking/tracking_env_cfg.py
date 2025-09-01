from __future__ import annotations

from dataclasses import MISSING  # 标记配置中必须手动设置的字段（未初始化）

import isaaclab.sim as sim_utils  # Isaac Lab仿真工具模块（灯光、材质、物理配置等）
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # 资产配置类：关节型机器人、基础资产（灯光等）
from isaaclab.envs import ManagerBasedRLEnvCfg  # 基于管理器的强化学习环境配置基类
from isaaclab.managers import (
    EventTermCfg as EventTerm,        # 事件配置类（如启动时随机化、定时扰动）
    ObservationGroupCfg as ObsGroup,  # 观测组配置类（按用途分组，如策略观测、特权观测）
    ObservationTermCfg as ObsTerm,    # 单个观测项配置类
    RewardTermCfg as RewTerm,        # 单个奖励项配置类
    SceneEntityCfg,                   # 场景实体配置类（指定资产/传感器/部件）
    TerminationTermCfg as DoneTerm    # 单个终止项配置类
)
from isaaclab.scene import InteractiveSceneCfg  # 交互式场景配置基类（包含地形、资产、传感器）
from isaaclab.sensors import ContactSensorCfg    # 接触传感器配置类（检测碰撞力、离地时间）
from isaaclab.terrains import TerrainImporterCfg # 地形导入配置类（平面、高度场等）

##
# 预定义配置模块导入
##
from isaaclab.utils import configclass  # 配置类装饰器（自动处理配置字段验证、序列化）
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # 加性均匀噪声配置类

# 导入自定义的机器人跟踪任务MDP模块（包含指令、动作、观测等函数）
import whole_body_tracking.tasks.tracking.mdp as mdp

##
# 场景定义：配置仿真环境中的地形、机器人、灯光、传感器
##

# 全局变量：机器人速度扰动范围（用于事件中的随机推斥）
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),    # x轴线速度扰动范围（m/s）
    "y": (-0.5, 0.5),    # y轴线速度扰动范围（m/s）
    "z": (-0.2, 0.2),    # z轴线速度扰动范围（m/s）
    "roll": (-0.52, 0.52),# 滚转角速度扰动范围（rad/s，≈30°/s）
    "pitch": (-0.52, 0.52),# 俯仰角速度扰动范围（rad/s）
    "yaw": (-0.78, 0.78)  # 偏航角速度扰动范围（rad/s，≈45°/s）
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """
    机器人跟踪任务的场景配置类（继承自交互式场景基类）
    功能：定义仿真环境中的地形、机器人、灯光、接触传感器等实体
    """

    # 1. 地面地形配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 地形在USD场景中的路径
        terrain_type="plane",        # 地形类型：平面（也支持heightfield等复杂地形）
        collision_group=-1,          # 碰撞组：-1表示默认组（与所有其他组碰撞）
        # 物理材质配置（影响摩擦、恢复系数）
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦组合模式：两物体摩擦系数相乘
            restitution_combine_mode="multiply",# 恢复系数组合模式：相乘
            static_friction=1.0,               # 静摩擦系数
            dynamic_friction=1.0               # 动摩擦系数
        ),
        # 视觉材质配置（USD模型外观）
        visual_material=sim_utils.MdlFileCfg(
            # 材质文件路径（使用NVIDIA Nucleus内置材质）
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,  # 启用UVW投影（确保材质正确映射到平面）
        ),
    )

    # 2. 机器人配置（关节型资产，具体机器人参数需在实例化时补充，故初始为MISSING）
    robot: ArticulationCfg = MISSING

    # 3. 灯光配置（确保仿真场景有足够光照，便于可视化和渲染）
    light = AssetBaseCfg(
        prim_path="/World/light",  # 平行光在USD中的路径
        # 平行光参数：冷白光（RGB 0.75）、强度3000流明
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",  # 穹顶光在USD中的路径
        # 穹顶光参数：弱环境光（RGB 0.13）、强度1000流明（模拟天空漫反射）
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    # 4. 接触传感器配置（检测机器人与环境的碰撞力，用于奖励/终止判断）
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # 传感器检测路径：所有环境的Robot下所有部件
        history_length=3,                     # 传感器数据历史长度（用于平滑判断）
        track_air_time=True,                  # 启用离地时间跟踪（判断脚部是否悬空）
        force_threshold=10.0,                 # 碰撞力阈值：超过10N才视为有效接触
        debug_vis=False                         # 启用可视化：在仿真中显示接触点
    )


##
# MDP配置：定义强化学习的核心模块（指令、动作、观测、奖励、终止、事件）
##


@configclass
class CommandsCfg:
    """
    MDP指令配置类：定义机器人需要跟踪的目标指令（此处为预加载的运动数据）
    """

    # 运动跟踪指令（使用自定义的MotionCommand，继承自Isaac Lab的CommandTerm）
    motion = mdp.MotionCommandCfg(
        asset_name="robot",                  # 关联的机器人资产名称
        resampling_time_range=(1.0e9, 1.0e9),# 指令重采样时间范围（极大值表示不主动重采样）
        debug_vis=True,                      # 启用指令可视化（显示目标锚点/部件）
        # 姿态随机扰动范围（增强训练鲁棒性，避免过拟合到固定运动）
        pose_range={
            "x": (-0.05, 0.05),    # x轴位置扰动（±5cm）
            "y": (-0.05, 0.05),    # y轴位置扰动（±5cm）
            "z": (-0.01, 0.01),    # z轴位置扰动（±1cm，小范围避免高度偏差过大）
            "roll": (-0.1, 0.1),   # 滚转姿态扰动（±0.1rad≈5.7°）
            "pitch": (-0.1, 0.1),  # 俯仰姿态扰动（±0.1rad）
            "yaw": (-0.2, 0.2)     # 偏航姿态扰动（±0.2rad≈11.5°）
        },
        velocity_range=VELOCITY_RANGE,       # 速度扰动范围（复用全局变量）
        joint_position_range=(-0.1, 0.1)     # 关节位置扰动范围（±0.1rad≈5.7°）
    )


@configclass
class ActionsCfg:
    """
    MDP动作配置类：定义机器人的控制输入（此处为关节位置控制）
    """

    # 关节位置动作（使用自定义的JointPositionAction，输出关节目标位置）
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",          # 关联的机器人资产名称
        joint_names=[".*"],          # 控制的关节：匹配所有关节（正则表达式".*"）
        use_default_offset=True      # 启用关节默认偏移：以机器人零位为基准
    )


@configclass
class ObservationsCfg:
    """
    MDP观测配置类：定义机器人的输入观测（分为策略观测和特权观测两组）
    策略观测：机器人"能感知到"的信息（带噪声，模拟真实传感器）
    特权观测：仅用于评估或辅助训练的真实信息（无噪声，如真实目标位置）
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """
        策略观测组：用于强化学习智能体的输入（带噪声，模拟真实环境感知）
        """

        # 1. 运动指令观测：获取目标运动指令（如关节目标位置/速度）
        command = ObsTerm(
            func=mdp.generated_commands,  # 自定义函数：获取生成的运动指令
            params={"command_name": "motion"}  # 参数：指令名称为"motion"
        )
        # # 2. 运动锚点相对位置观测（机器人锚点坐标系下）
        # motion_anchor_pos_b = ObsTerm(
        #     func=mdp.motion_anchor_pos_b,  # 自定义函数：计算目标锚点相对机器人锚点的位置
        #     params={"command_name": "motion"},  # 参数：关联"motion"指令
        #     noise=Unoise(n_min=-0.25, n_max=0.25)  # 加性噪声：±0.25m（模拟位置传感器噪声）
        # )
        # # 3. 运动锚点相对姿态观测（机器人锚点坐标系下）
        # motion_anchor_ori_b = ObsTerm(
        #     func=mdp.motion_anchor_ori_b,  # 自定义函数：计算目标锚点相对机器人锚点的姿态
        #     params={"command_name": "motion"},
        #     noise=Unoise(n_min=-0.05, n_max=0.05)  # 加性噪声：±0.05rad（模拟姿态传感器噪声）
        # )
        # 4. 机器人基座线速度观测（世界坐标系下）
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,  # 自定义函数：获取机器人基座线速度
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 加性噪声：±0.5m/s（模拟IMU线速度噪声）
        )
        # 5. 机器人基座角速度观测（世界坐标系下）
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,  # 自定义函数：获取机器人基座角速度
            noise=Unoise(n_min=-0.2, n_max=0.2)  # 加性噪声：±0.2rad/s（模拟IMU角速度噪声）
        )
        # 6. 关节相对位置观测（相对于零位的偏差）
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,  # 自定义函数：获取关节相对零位的位置
            noise=Unoise(n_min=-0.03, n_max=0.03)  # 加性噪声：±0.01rad（模拟关节编码器噪声）
        )
        # 7. 关节相对速度观测
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,  # 自定义函数：获取关节相对速度
            noise=Unoise(n_min=-1.5, n_max=1.5)  # 加性噪声：±0.5rad/s（模拟关节速度传感器噪声）
        )
        # 8. 上一时刻动作观测（用于动作平滑性约束）
        actions = ObsTerm(func=mdp.last_action)  # 自定义函数：获取上一时刻的动作

        def __post_init__(self):
            """配置后初始化：设置观测组的全局属性"""
            self.enable_corruption = True  # 启用观测损坏（噪声、延迟等，增强鲁棒性）
            self.concatenate_terms = True  # 将所有观测项拼接为一个向量（策略输入需一维向量）

    @configclass
    class PrivilegedCfg(ObsGroup):
        """
        特权观测组：用于评估或辅助训练（如BC训练），无噪声，提供真实环境信息
        """
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})  # 机器人部件真实位置
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})  # 机器人部件真实姿态
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # 观测组实例化：策略观测和特权观测
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()  # 批评家（Critic）使用特权观测评估价值


@configclass
class EventCfg:
    """
    MDP事件配置类：定义环境中的动态事件（启动时随机化、定时扰动）
    作用：增强训练多样性，避免过拟合，提升机器人鲁棒性
    """

    # 1. 启动时事件：随机化机器人刚体材质（摩擦系数等）
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # 自定义函数：随机化刚体材质
        mode="startup",                          # 事件模式：仅在环境启动时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 目标资产：机器人所有部件
            "static_friction_range": (0.3, 1.6),  # 静摩擦系数随机范围
            "dynamic_friction_range": (0.3, 1.2), # 动摩擦系数随机范围
            "restitution_range": (0.0, 0.5),      # 恢复系数随机范围（0=完全非弹性，0.5=半弹性）
            "num_buckets": 64                     # 分桶数量：减少随机性波动（按桶采样）
        },
    )

    # 2. 启动时事件：随机化关节默认位置（添加小偏移）
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,  # 自定义函数：随机化关节默认位置
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),  # 目标关节：所有关节
            "pos_distribution_params": (-0.01, 0.01),  # 位置偏移范围：±0.01rad
            "operation": "add"  # 操作类型：在默认位置基础上叠加偏移
        },
    )

    # 3. 启动时事件：随机化机器人躯干质心（COM）
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,  # 自定义函数：随机化刚体质心
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),  # 目标部件：躯干
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},  # 质心偏移范围
        },
        # params={
        #     "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),  # 目标部件：躯干
        #     "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},  # 质心偏移范围
        # },
    )

    # 4. 间隔事件：定时推斥机器人（模拟外部扰动，增强抗干扰能力）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,  # 自定义函数：通过设置速度推斥机器人
        mode="interval",                    # 事件模式：按时间间隔执行
        interval_range_s=(1.0, 3.0),        # 执行间隔：1~3秒随机一次
        params={"velocity_range": VELOCITY_RANGE},  # 推斥速度范围（复用全局变量）
    )



@configclass
class RewardsCfg:
    """
    MDP奖励配置类：定义强化学习的奖励函数集合，用于引导机器人完成运动跟踪任务
    奖励设计逻辑：以"减小跟踪误差"为核心，辅以"约束不良行为"，通过权重调节各目标优先级
    """

    # 1. 全局锚点位置跟踪奖励：基于锚点（如躯干）与目标位置的误差，采用指数衰减函数
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,  # 自定义奖励函数：锚点位置误差的指数衰减
        weight=0.5,  # 奖励权重（0.5，优先级低于部件跟踪）
        params={
            "command_name": "motion",  # 关联的运动指令名称（对应CommandsCfg中的motion）
            "std": 0.3  # 误差标准差（控制奖励衰减速度，std越小，误差对奖励影响越敏感）
        },
    )

    # 2. 全局锚点姿态跟踪奖励：基于锚点与目标姿态的误差，指数衰减
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,  # 自定义函数：锚点姿态误差的指数衰减
        weight=0.5,  # 权重0.5，与锚点位置奖励优先级一致
        params={"command_name": "motion", "std": 0.4}  # std=0.4，姿态误差的衰减速度略慢于位置
    )

    # 3. 部件相对位置跟踪奖励：基于机器人各部件与目标部件的相对位置误差，指数衰减
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,  # 自定义函数：部件相对位置误差的指数衰减
        weight=1.0,  # 权重1.0（优先级高于锚点，部件跟踪是精细控制核心）
        params={"command_name": "motion", "std": 0.3}  # std=0.3，与锚点位置误差的敏感程度一致
    )

    # 4. 部件相对姿态跟踪奖励：基于机器人各部件与目标部件的相对姿态误差，指数衰减
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,  # 自定义函数：部件相对姿态误差的指数衰减
        weight=1.0,  # 权重1.0，与部件位置奖励优先级一致
        params={"command_name": "motion", "std": 0.4}  # std=0.4，姿态误差衰减速度略慢
    )

    # 5. 部件全局线速度跟踪奖励：基于机器人部件与目标部件的全局线速度误差，指数衰减
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,  # 自定义函数：部件线速度误差的指数衰减
        weight=1.0,  # 权重1.0，速度跟踪与位置/姿态跟踪同等重要
        params={"command_name": "motion", "std": 1.0}  # std=1.0，速度误差的衰减速度较慢（允许更大速度波动）
    )

    # 6. 部件全局角速度跟踪奖励：基于机器人部件与目标部件的全局角速度误差，指数衰减
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,  # 自定义函数：部件角速度误差的指数衰减
        weight=1.0,  # 权重1.0，角速度跟踪是姿态平滑的关键
        params={"command_name": "motion", "std": 3.14}  # std=3.14（≈π），角速度误差允许范围较大
    )

    # 7. 动作平滑性惩罚：基于动作的L2范数（动作变化率），负奖励惩罚剧烈动作
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,  # 自定义函数：计算动作变化率的L2范数（当前动作与上一动作的差）
        weight=-1e-1  # 负权重-0.1（轻微惩罚，避免动作突变导致机器人不稳定）
    )

    # 8. 关节限位惩罚：当关节位置接近物理极限时，给予强负奖励
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,  # 自定义函数：检测关节位置是否接近限位，计算惩罚值
        weight=-10.0,  # 负权重-10.0（强惩罚，防止关节超限位导致机械损坏或仿真错误）
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])  # 目标资产：机器人所有关节（正则".*"匹配所有）
        },
    )

    # 9. 非期望接触惩罚：当非接触部件（如躯干、手臂）与地面碰撞时，给予负奖励
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,  # 自定义函数：检测非期望接触并计算惩罚
        weight=-0.1,  # 负权重-0.1（轻微惩罚，允许小概率碰撞，避免过度约束）
        # params={
        #     "sensor_cfg": SceneEntityCfg(
        #         "contact_forces",  # 关联的接触传感器名称（对应MySceneCfg中的contact_forces）
        #         # 排除的接触部件（仅允许这些部件接触地面：左右脚踝滚转关节、左右手腕偏航关节）
        #         body_names=[
        #             r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
        #         ],
        #     ),
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",  # 关联的接触传感器名称（对应MySceneCfg中的contact_forces）
                # 排除的接触部件（仅允许这些部件接触地面：左右脚踝滚转关节、左右手腕偏航关节）
                # body_names=[
                #     r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                # ],
                body_names=[
                    r"^(?!L_ankle_pitch_link$)(?!R_ankle_pitch_link$).+$"
                ],
            ),
            "threshold": 1.0  # 接触力阈值：超过1N的接触才视为有效非期望接触
        },
    )


@configclass
class TerminationsCfg:
    """
    MDP终止配置类：定义episode（训练回合）的终止条件，用于判断任务失败或结束
    终止逻辑：当机器人严重偏离目标或超时，立即终止回合，避免无效训练
    """

    # 1. 超时终止：当回合持续时间达到设定上限时终止
    time_out = DoneTerm(
        func=mdp.time_out,  # 自定义函数：判断是否达到episode时间上限
        time_out=True  # 启用超时终止（对应TrackingEnvCfg中的episode_length_s）
    )

    # 2. 锚点高度偏差终止：锚点（如躯干）的Z轴（高度）偏差超过阈值时终止
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,  # 自定义函数：仅判断锚点Z轴位置偏差是否超标
        params={
            "command_name": "motion",  # 关联的运动指令
            "threshold": 0.25  # 高度偏差阈值：超过0.25m（25cm）视为严重偏离，终止回合
        },
    )

    # 3. 锚点姿态偏差终止：锚点姿态偏差超过阈值时终止
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,  # 自定义函数：判断锚点姿态偏差是否超标
        params={
            "asset_cfg": SceneEntityCfg("robot"),  # 目标资产：机器人
            "command_name": "motion",  # 关联运动指令
            "threshold": 0.8  # 姿态偏差阈值：超过0.8视为严重倾斜（如躯干过度前倾/后仰），终止回合
        },
    )

    # 4. 末端执行器高度偏差终止：手脚末端部件的Z轴偏差超过阈值时终止
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,  # 自定义函数：仅判断指定部件Z轴位置偏差是否超标
        # params={
        #     "command_name": "motion",  # 关联运动指令
        #     "threshold": 0.25,  # 高度偏差阈值：25cm（与锚点高度阈值一致）
        #     "body_names": [  # 需要检测的末端部件：左右脚踝、左右手腕
        #         "left_ankle_roll_link",
        #         "right_ankle_roll_link",
        #         "left_wrist_yaw_link",
        #         "right_wrist_yaw_link",
        #     ],
        # },
        params={
            "command_name": "motion",  # 关联运动指令
            "threshold": 0.25,  # 高度偏差阈值：25cm（与锚点高度阈值一致）
            "body_names": [  # 需要检测的末端部件：左右脚踝、左右手腕
                "L_ankle_pitch_link",
                "R_ankle_pitch_link",
            ],
        },
    )


@configclass
class CurriculumCfg:
    """
    MDP课程学习配置类：用于逐步提升训练难度（如增加扰动范围、复杂地形）
    当前配置为"pass"，表示暂不启用课程学习，所有训练难度保持一致
    后续可扩展：如随着训练迭代增加，扩大姿态扰动范围、引入高度场地形等
    """

    pass


##
# 环境主配置：整合所有子配置，定义强化学习环境的核心参数
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """
    运动跟踪任务的强化学习环境主配置类（继承自Isaac Lab的ManagerBasedRLEnvCfg）
    功能：整合场景、观测、动作、指令、奖励、终止等所有子配置，定义环境全局参数
    """

    # 1. 场景配置：关联自定义的MySceneCfg，设置多环境数量和间距
    scene: MySceneCfg = MySceneCfg(
        num_envs=4096,  # 并行训练的环境数量（4096，利用GPU并行加速训练）
        env_spacing=2.5  # 环境间的间距（2.5m，避免不同环境的机器人碰撞）
    )

    # 2. 基础MDP配置：关联观测、动作、指令子配置
    observations: ObservationsCfg = ObservationsCfg()  # 观测配置（策略观测、特权观测）
    actions: ActionsCfg = ActionsCfg()                  # 动作配置（关节位置控制）
    commands: CommandsCfg = CommandsCfg()               # 指令配置（运动跟踪指令）

    # 3. MDP核心配置：关联奖励、终止、事件、课程学习子配置
    rewards: RewardsCfg = RewardsCfg()                  # 奖励函数集合
    terminations: TerminationsCfg = TerminationsCfg()  # 终止条件集合
    events: EventCfg = EventCfg()                       # 动态事件集合（随机化、扰动）
    curriculum: CurriculumCfg = CurriculumCfg()         # 课程学习配置（暂不启用）

    def __post_init__(self):
        """
        配置后初始化：在配置类实例化后自动执行，设置环境全局参数
        作用：补充主配置中未显式定义的参数，确保环境正常运行
        """
        # ------------------------------ 通用训练参数 ------------------------------
        self.decimation = 4  # 动作下采样率：每4个仿真步执行一次动作（降低训练计算量）
        self.episode_length_s = 10.0  # 每个回合的持续时间（10秒，超时后终止回合）

        # ------------------------------ 仿真参数 ------------------------------
        self.sim.dt = 0.005  # 仿真步长（0.005秒，即200Hz，平衡仿真精度与速度）
        self.sim.render_interval = self.decimation  # 渲染间隔：与动作下采样率一致（每4步渲染一次）
        self.sim.physics_material = self.scene.terrain.physics_material  # 仿真物理材质：复用地形的材质配置
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # PhysX GPU刚体补丁数量上限（避免仿真卡顿）

        # ------------------------------ 可视化viewer参数 ------------------------------
        self.viewer.eye = (1.5, 1.5, 1.5)  # viewer相机位置（x=1.5, y=1.5, z=1.5，斜上方视角）
        self.viewer.origin_type = "asset_root"  # 相机参考原点类型：以机器人资产的根节点为参考
        self.viewer.asset_name = "robot"  # 参考资产名称：相机跟随机器人运动