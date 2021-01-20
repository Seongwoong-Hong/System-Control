import gym
import torch as th
import numpy as np
from scipy import linalg
from stable_baselines3.common.policies import BasePolicy

class LQRPolicy(BasePolicy):
    def __init__(self, env):
        observation_space = env.observation_space
        action_space = env.action_space
        super(LQRPolicy, self).__init__(
            observation_space,
            action_space)

        self.gear = env.model.actuator_gear[0, 0]
        self.P, self.D = self._get_gains()

    def _get_gains(self):
        A, B, Q, R = self._build_env()
        X = linalg.solve_continuous_are(A, B, Q, R)
        K = (1/R * (B.T @ X)).reshape(-1)
        return K[1], K[0]

    def _build_env(self):
        raise NotImplementedError

    def forward(self):
        return None

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return -1/self.gear * (self.P*observation[0][1] + self.D*observation[0][0]).reshape(1, 1)