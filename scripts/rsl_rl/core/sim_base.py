import math
import time
import torch
import numpy as np
from collections import deque


class NanoSleep:
    def __init__(self, ms: float):
        self.duration_sec = ms * 1e-3

    def waiting(self, start_time: float):
        while time.perf_counter() - start_time < self.duration_sec:
            pass


class SimBase:
    def __init__(self, cfg ):
        self.cfg = cfg

        self.action_dim = cfg.env.num_actions
        self.single_obs_dim = cfg.env.num_single_obs
        self.frame_stack = cfg.env.frame_stack

        # 状态变量初始化
        self.action = np.zeros(self.action_dim, dtype=np.float32)
        self.target_q = np.zeros(self.action_dim, dtype=np.float32)

        # 历史观测缓存
        self.hist_obs = deque([
            np.zeros([1, self.single_obs_dim], dtype=np.float32)
            for _ in range(self.frame_stack)
        ], maxlen=self.frame_stack)

        self.hist_obs_run = deque([
            np.zeros([1, self.single_obs_dim + 2], dtype=np.float32)
            for _ in range(self.frame_stack)
        ], maxlen=self.frame_stack)

    def get_action(self, obs: np.ndarray, mode: str = "walk") -> np.ndarray:
        """
        通用接口，mode 支持 'walk' 和 'run'，根据策略选择行为
        """
        if mode == "walk":
            return self._process_action(
                obs, self.hist_obs, self.policy_walk,
                input_dim=self.single_obs_dim,
                pos0=self.cfg.real_config.pos0[:10]
            )
        elif mode == "run":
            return self._process_action(
                obs, self.hist_obs_run, self.policy_run,
                input_dim=self.single_obs_dim + 2,
                pos0=self.cfg.real_config.turn_pos0[:10]
            )
        else:
            raise ValueError(f"[SimBase] Unsupported mode '{mode}'")

    def _process_action(self, obs: np.ndarray, buffer: deque, policy, input_dim: int, pos0: np.ndarray) -> np.ndarray:

        buffer.append(obs)
        stacked_obs = np.concatenate([o[0] for o in buffer], axis=0).reshape(1, -1)
        with torch.no_grad():
            action = policy(torch.tensor(stacked_obs, dtype=torch.float32))[0].cpu().numpy()
        action = np.clip(action, -18.0, 18.0)
        self.action = action
        self.target_q = action * self.cfg.control.action_scale
        return self.target_q + pos0
