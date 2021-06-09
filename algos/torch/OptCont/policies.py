import torch as th
import numpy as np
from scipy import linalg
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv


class LQRPolicy(BasePolicy):
    def __init__(self, env, noise_lv: float = 0.1,
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

        self.env = env
        if isinstance(env, DummyVecEnv):
            self.env = env.envs[0]
        self.noise_lv = noise_lv
        self._build_env()
        self.K = self._get_gains()

    def _get_gains(self) -> th.Tensor:
        X = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        if self.R.shape[0] >= 2:
            K = (np.linalg.inv(self.R) @ (self.B.T @ X))
        else:
            K = (1/self.R * (self.B.T @ X)).reshape(-1)
        return th.from_numpy(K)

    def _build_env(self) -> np.array:
        # define self.A, self.B, self.Q, self.R
        # Their types are all numpy array even though they are 1-dim value.
        raise NotImplementedError

    def forward(self):
        return None

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        noise = 0
        if not deterministic:
            noise = self.noise_lv * np.random.randn(*self.K.shape)

        return -1 / self.gear * ((self.K + noise) @ observation.T).reshape(1, -1)
