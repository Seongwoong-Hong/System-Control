import torch as th
import numpy as np
from scipy import linalg
from typing import Optional, Tuple
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


class LQRPolicy(BasePolicy):
    def __init__(
            self,
            env,
            noise_lv: float = 0.1,
            observation_space=None,
            action_space=None,
    ):
        if observation_space is None:
            observation_space = env.observation_space
        if action_space is None:
            action_space = env.action_space
        super(LQRPolicy, self).__init__(observation_space, action_space)

        self.env = None
        self.set_env(env)
        self.noise_lv = noise_lv
        self._build_env()
        self._get_gains()

    def set_env(self, env):
        if not isinstance(env, VecEnv):
            self.env = DummyVecEnv([lambda: env])
        else:
            self.env = env

    def _get_gains(self):
        X = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        if self.R.shape[0] >= 2:
            K = (np.linalg.inv(self.R) @ (self.B.T @ X))
        else:
            K = (1 / self.R * (self.B.T @ X)).reshape(-1)
        self.K = th.from_numpy(K).double()

    def _build_env(self):
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


class DiscreteLQRPolicy(LQRPolicy):
    def _build_env(self):
        # define self.A, self.B, self.Q, self.R
        # Their types are all numpy array even though they are 1-dim value.
        raise NotImplementedError

    def _get_gains(self):
        X = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        if self.R.shape[0] >= 2:
            K = (np.linalg.inv(self.B.T @ X @ self.B + self.R) @ (self.B.T @ X))
        else:
            K = (1 / (self.B.T @ X @ self.B + self.R) * (self.B.T @ X)).reshape(-1)
        self.K = th.from_numpy(K).double()


class FiniteLQRPolicy(LQRPolicy):
    def _build_env(self):
        # define self.A, self.B, self.Q, self.R
        # Their types are all numpy array even though they are 1-dim value.
        raise NotImplementedError

    def _get_gains(self):
        self.max_t = self.env.get_attr("spec")[0].max_episode_steps
        self.q_series = np.zeros_like(self.Q)[None, ...].repeat(self.max_t + 1, axis=0)
        self.k_series = np.zeros_like(self.B.T)[None, ...].repeat(self.max_t, axis=0)
        self.sigma_series = np.zeros_like(self.R)[None, ...].repeat(self.max_t, axis=0)
        self.q_series[-1] = self.Q
        for t in reversed(range(self.max_t)):
            self.sigma_series[t] = np.linalg.inv(self.R + self.B.T @ self.q_series[t + 1] @ self.B)
            self.k_series[t] = -self.sigma_series[t] @ self.B.T @ self.q_series[t + 1] @ self.A
            self.q_series[t] = self.Q + self.A.T @ self.q_series[t + 1] @ self.A - \
                               self.A.T @ self.q_series[t + 1] @ self.B @ (- self.k_series[t])

    def learn(self, *args, **kwargs):

        self._get_gains()

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        obs_list, act_list, rew_list = [], [], []
        self.env.reset()
        self.env.env_method("set_state", observation.squeeze())
        obs_list.append(observation.squeeze())
        for t in range(self.max_t):
            act = self.k_series[t] @ observation.reshape(-1, 1)
            if not deterministic:
                eps = np.random.standard_normal(self.env.action_space.shape).reshape(-1, 1)
                act += np.linalg.cholesky(self.sigma_series[t]) @ eps
                act = np.clip(act, a_min=-100, a_max=100)
            observation, reward, done, info = self.env.step(act.reshape(1, -1) / 100)
            if done:
                observation = info[0]['terminal_observation']
            obs_list.append(observation.flatten())
            rew_list.append(reward.flatten())
            act_list.append(act.flatten())
        return np.array(obs_list), np.array(act_list), np.array(rew_list)
