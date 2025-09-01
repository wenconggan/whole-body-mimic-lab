from isaaclab.utils import configclass

from whole_body_tracking.robots.x2 import x2_ACTION_SCALE, x2_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.x2.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class x2FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = x2_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = x2_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis_link"
        self.commands.motion.body_names = [
            "pelvis_link",
            "L_hip_roll_link",
            "L_knee_pitch_link",
            "L_ankle_pitch_link",
            "R_hip_roll_link",
            "R_knee_pitch_link",
            "R_ankle_pitch_link",
            "L_shoulder_roll_link",
            "L_elbow_pitch_link",
            "R_shoulder_roll_link",
            "R_elbow_pitch_link",
        ]


@configclass
class x2FlatWoStateEstimationEnvCfg(x2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class x2FlatLowFreqEnvCfg(x2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
