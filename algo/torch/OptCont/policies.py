import torch as th
import numpy as np
from scipy import linalg
from stable_baselines3.common.policies import BasePolicy


class LQRPolicy(BasePolicy):
    def __init__(self, env, noise_lv: float = 0.25,
                 observation_space=None,
                 action_space=None,
                 ):
        if observation_space is None:
            observation_space = env.observation_space
        if action_space is None:
            action_space = env.action_space
        super(LQRPolicy, self).__init__(
            observation_space,
            action_space)

        self.noise_lv = noise_lv
        self.gear = env.model.actuator_gear[0, 0]
        self.K = self._get_gains()

    def _get_gains(self) -> th.Tensor:
        A, B, Q, R = self._build_env()
        X = linalg.solve_continuous_are(A, B, Q, R)
        if R.shape[0] >= 2:
            K = (np.linalg.inv(R) @ (B.T @ X))
        else:
            K = (1/R * (B.T @ X)).reshape(-1)
        return th.from_numpy(K)

    def _build_env(self) -> np.array:
        # returns A, B, Q, R
        # Their types are all numpy array even though they are 1-dim value.
        raise NotImplementedError

    def forward(self):
        return None

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        if not deterministic:
            self.K += self.noise_lv * np.random.randn(*self.K.shape)

        return -1/self.gear * (self.K @ observation.T).reshape(1, -1)
