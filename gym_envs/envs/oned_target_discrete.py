import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List


class OneDTargetDisc(gym.Env):
    def __init__(self):
        self.dt = 0.1
        self.timesteps = 0
        self.st = None
        self.observation_space = gym.spaces.MultiDiscrete([100])
        self.action_space = gym.spaces.MultiDiscrete([3])

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        r = self.reward_fn(self.st, action)
        self.st = self.st + action - 1  # + 0.03 * np.random.random(self.st.shape)
        if self.st[0] < 0:
            self.st[0] += 1
        if self.st[0] >= self.observation_space.nvec[0]:
            self.st[0] -= 1
        self.timesteps += 1
        return self.st, r, None, info

    def _get_obs(self):
        return self.st

    def set_state(self, obs: np.ndarray):
        assert self.observation_space.nvec[0] > obs.item() >= 0
        self.st = obs

    def reset(self):
        self.set_state(self.observation_space.sample())
        self.timesteps = 0
        return self._get_obs()

    def reward_fn(self, state, action) -> float:
        x = state[0] + action[0] - 1
        return - ((x - 80) ** 2)

    def render(self, mode='human'):
        pass


class OneDTargetDiscDet(OneDTargetDisc):
    def __init__(self):
        super(OneDTargetDiscDet, self).__init__()
        self.n = 0

    def reset(self):
        self.st = np.array([self.n % self.observation_space.nvec[0]], dtype=int)
        self.timesteps = 0.0
        self.n += 1
        return self._get_obs()

