import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##从 Gym 注册表加载配置  load_cfg_from_registry

# 使用 Gymnasium 的 gym.register() 函数注册 Isaac Lab 自定义强化学习环境
# 作用：将环境纳入 Gym 生态，后续可通过 gym.make("环境ID") 快速创建环境实例
gym.register(
    # 1. 环境唯一标识符（ID）：遵循 Gymnasium 命名规范，格式清晰易区分
    # 结构解析：[任务类型]-[环境地形]-[机器人型号]-[版本号]
    # Tracking：任务类型为“轨迹跟踪”（如跟踪预设运动轨迹）
    # Flat：环境地形为“平坦地面”（无障碍物、坡度）
    # x2：使用的机器人型号为“x2”（如四足机器人x2）
    # v0：环境版本号（后续修改环境参数时可升级版本，避免冲突）
    id="x2_mimic",

    # 2. 环境类入口点：指定该环境的核心实现类
    # 格式："模块路径:类名"，即从 isaaclab.envs 模块中导入 ManagerBasedRLEnv 类
    # ManagerBasedRLEnv：Isaac Lab 提供的“基于管理器的RL环境基类”
    # 特点：支持多环境并行仿真、物理引擎高效交互、传感器数据统一管理
    entry_point="isaaclab.envs:ManagerBasedRLEnv",

    # 3. 禁用 Gym 内置环境检查器
    # 原因：Isaac Lab 环境已通过自身机制（如配置类校验、物理参数检查）确保合规性
    # 禁用默认检查器可避免重复校验，提升环境初始化速度
    disable_env_checker=True,

    # 4. 传递给环境构造函数（ManagerBasedRLEnv.__init__）的关键字参数
    # 核心作用：注入环境配置和RL算法配置，实现“配置与代码解耦”
    kwargs={
        # 4.1 环境配置入口：指定该环境的具体参数配置类
        # flat_env_cfg：提前定义的配置模块（如包含场景、机器人、奖励函数的配置）
        # x2FlatEnvCfg：该模块中针对“x2机器人平坦地形跟踪任务”的配置类
        # 内容包括：并行环境数量、机器人USD路径、观测/动作空间定义、奖励系数等
        "env_cfg_entry_point": flat_env_cfg.x2FlatEnvCfg,

        # 4.2 RSL-RL算法配置入口：指定该任务使用的RL算法（PPO）参数配置
        # f-string 动态生成路径：确保模块名变化时仍能正确引用
        # agents.__name__：获取 agents 模块的名称（如 "whole_body_tracking.agents"）
        # rsl_rl_ppo_cfg：RSL-RL库中PPO算法的配置模块
        # x2FlatPPORunnerCfg：针对“x2机器人平坦地形任务”的PPO运行器配置类
        # 内容包括：学习率、批大小、策略网络结构、训练迭代次数等
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:x2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-x2-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.x2FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:x2FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-x2-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.x2FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:x2FlatLowFreqPPORunnerCfg",
    },
)
