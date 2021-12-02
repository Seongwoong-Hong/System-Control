import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List


class OneDTargetDisc(gym.Env):
    def __init__(self, map_size=10):
        self.dt = 0.1
        self.timesteps = 0
        self.st = None
        self.map_size = map_size
        self.observation_space = gym.spaces.MultiDiscrete([self.map_size])
        self.action_space = gym.spaces.MultiDiscrete([3])

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        ft = np.zeros(self.map_size)
        ft[self.st] = 1
        info = {'obs': ft.reshape(1, -1), 'acts': action.reshape(1, -1)}
        r = self.reward_fn(self.st, action)
        self.st = self.st + action - 1  # + 0.03 * np.random.random(self.st.shape)
        if self.st[0] < 0:
            self.st[0] += 1
        if self.st[0] >= self.map_size:
            self.st[0] -= 1
        self.timesteps += 1
        return self.st, r, None, info

    def _get_obs(self):
        return self.st

    def set_state(self, obs: np.ndarray):
        assert self.map_size > obs.item() >= 0
        self.st = obs

    def reset(self):
        self.set_state(self.observation_space.sample())
        self.timesteps = 0
        return self._get_obs()

    def reward_fn(self, state, action) -> float:
        x = state[0]
        return - ((x - 15) ** 2)

    def render(self, mode='human'):
        pass


class OneDTargetDiscDet(OneDTargetDisc):
    def __init__(self, map_size=10):
        super(OneDTargetDiscDet, self).__init__(map_size=map_size)
        self.n = 0

    def reset(self):
        self.st = np.array([self.n % self.map_size], dtype=int)
        self.timesteps = 0
        self.n += 1
        return self._get_obs()

