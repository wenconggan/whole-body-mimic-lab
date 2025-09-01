import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# 各型号电机的转动惯量参数
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# 控制器参数配置
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz的自然频率(转换为弧度)
DAMPING_RATIO = 2.0  # 阻尼比

# 根据转动惯量和自然频率计算各型号电机的刚度
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ ** 2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ ** 2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ ** 2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ ** 2

# 根据转动惯量、阻尼比和自然频率计算各型号电机的阻尼
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# x2机器人模型配置
x2_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,  # 不固定机器人基座
        replace_cylinders_with_capsules=True,  # 将圆柱体替换为胶囊体(提高碰撞检测性能)
        asset_path=f"{ASSET_DIR}/x2/x2.urdf",  # URDF模型路径
        activate_contact_sensors=True,  # 启用接触传感器
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 启用重力
            retain_accelerations=False,
            linear_damping=0.0,  # 线性阻尼
            angular_damping=0.0,  # 角阻尼
            max_linear_velocity=1000.0,  # 最大线速度
            max_angular_velocity=1000.0,  # 最大角速度
            max_depenetration_velocity=1.0,  # 最大分离速度
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,  # 启用自碰撞
            solver_position_iteration_count=8,  # 位置求解器迭代次数
            solver_velocity_iteration_count=4  # 速度求解器迭代次数
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)  # 关节驱动增益(初始化为0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),  # 初始位置(x, y, z)
        joint_pos={  # 初始关节角度
            ".*_hip_pitch": -0.3,
            ".*_knee_pitch": 0.6,
            ".*_ankle_pitch": -0.3,
            ".*_elbow_pitch": 0.0,
            "L_shoulder_roll": 0.0,
            "L_shoulder_pitch": 0.0,
            "R_shoulder_pitch": -0.0,
            "R_shoulder_roll": 0.0,
        },
        joint_vel={".*": 0.0},  # 初始关节速度(全部为0)
    ),
    soft_joint_pos_limit_factor=0.9,  # 关节位置软限制系数
    actuators={
        # 腿部执行器配置
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[  # 匹配的关节名称表达式
                ".*_hip_yaw",
                ".*_hip_roll",
                ".*_hip_pitch",
                ".*_knee_pitch",
                ".*_ankle_pitch"

],
            effort_limit_sim={  # 仿真中的力限制
                ".*_hip_yaw": 100.0,
                ".*_hip_roll": 100.0,
                ".*_hip_pitch": 100.0,
                ".*_knee_pitch": 100.0,
                ".*_ankle_pitch": 80,

            },
            velocity_limit_sim={  # 仿真中的速度限制
                ".*_hip_yaw": 30.0,
                ".*_hip_roll": 30.0,
                ".*_hip_pitch": 30.0,
                ".*_knee_pitch": 30.0,
                ".*_ankle_pitch": 30.0,

            },
            stiffness={  # 刚度配置
                ".*_hip_yaw": 200,
                ".*_hip_roll": 200,
                ".*_hip_pitch": 200,
                ".*_knee_pitch": 200,
                ".*_ankle_pitch": 30,

            },
            damping={  # 阻尼配置
                ".*_hip_yaw": 3,
                ".*_hip_roll": 3,
                ".*_hip_pitch": 3,
                ".*_knee_pitch": 3,
                ".*_ankle_pitch": 2,
            },
            armature={  # 转动惯量配置
                ".*_hip_pitch": ARMATURE_7520_14,
                ".*_hip_roll": ARMATURE_7520_22,
                ".*_hip_yaw": ARMATURE_7520_14,
                ".*_knee_pitch": ARMATURE_7520_22,
            },
        ),
        # 脚部执行器配置
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,  # 力限制
            velocity_limit_sim=30.0,  # 速度限制
            joint_names_expr=[".*_ankle_pitch"],  # 匹配的关节
            stiffness=30 ,
            damping=2,
            armature=2.0 * ARMATURE_5020,  # 转动惯量(5020型号的2倍)
        ),

        "arms": ImplicitActuatorCfg(
            joint_names_expr=[  # 匹配的关节名称表达式
                ".*_shoulder_pitch",
                ".*_shoulder_roll",
                ".*_shoulder_yaw",
                ".*_elbow_pitch",
                # ".*_wrist_roll_joint",
                # ".*_wrist_pitch_joint",
                # ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={  # 仿真中的力限制
                ".*_shoulder_pitch": 25.0,
                ".*_shoulder_roll": 25.0,
                ".*_shoulder_yaw": 25.0,
                ".*_elbow_pitch": 25.0,
                # ".*_wrist_roll_joint": 25.0,
                # ".*_wrist_pitch_joint": 5.0,
                # ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={  # 仿真中的速度限制
                ".*_shoulder_pitch": 37.0,
                ".*_shoulder_roll": 37.0,
                ".*_shoulder_yaw": 37.0,
                ".*_elbow_pitch": 37.0,
                # ".*_wrist_roll_joint": 37.0,
                # ".*_wrist_pitch_joint": 22.0,
                # ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={  # 刚度配置
                ".*_shoulder_pitch": 40,
                ".*_shoulder_roll": 40,
                ".*_shoulder_yaw": 40,
                ".*_elbow_pitch": 40,
                # ".*_wrist_roll_joint": STIFFNESS_5020,
                # ".*_wrist_pitch_joint": STIFFNESS_4010,
                # ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={  # 阻尼配置
                ".*_shoulder_pitch": 3,
                ".*_shoulder_roll": 3,
                ".*_shoulder_yaw": 3,
                ".*_elbow_pitch": 3,
                # ".*_wrist_roll_joint": DAMPING_5020,
                # ".*_wrist_pitch_joint": DAMPING_4010,
                # ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={  # 转动惯量配置
                ".*_shoulder_pitch": ARMATURE_5020,
                ".*_shoulder_roll": ARMATURE_5020,
                ".*_shoulder_yaw": ARMATURE_5020,
                ".*_elbow_pitch": ARMATURE_5020,
                # ".*_wrist_roll_joint": ARMATURE_5020,
                # ".*_wrist_pitch_joint": ARMATURE_4010,
                # ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
)

# 计算x2机器人的动作缩放因子
x2_ACTION_SCALE = {}
for a in x2_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim  # 力限制
    s = a.stiffness  # 刚度
    names = a.joint_names_expr  # 关节名称表达式

    # 如果力限制不是字典格式，则转换为字典(所有关节使用相同值)
    if not isinstance(e, dict):
        e = {n: e for n in names}
    # 如果刚度不是字典格式，则转换为字典(所有关节使用相同值)
    if not isinstance(s, dict):
        s = {n: s for n in names}

    # 计算每个关节的动作缩放因子
    for n in names:
        if n in e and n in s and s[n]:
            # 缩放因子 = 0.25 * 力限制 / 刚度
            x2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]