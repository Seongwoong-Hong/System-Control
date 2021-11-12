import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List


class OneDTargetDisc(gym.Env):
    def __init__(self):
        self.dt = 0.1
        self.st = None
        self.timesteps = 0.0
        self.observation_space = gym.spaces.MultiDiscrete([11])
        self.action_space = gym.spaces.MultiDiscrete([3])

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        r = self.reward_fn(self.st, action)
        self.st = self.st + action - 1  # + 0.03 * np.random.random(self.st.shape)
        done = bool(
            self.st[0] < 0
            or self.st[0] > 10
        )
        if done:
            r -= 1000
        info['done'] = done
        self.timesteps += self.dt
        return self.st, r, done, info

    def _get_obs(self):
        return self.st

    def set_state(self, obs: np.ndarray):
        assert 10 >= obs.item() >= 0
        self.st = obs

    def reset(self):
        self.st = self.observation_space.sample()
        self.timesteps = 0.0
        return self._get_obs()

    def reward_fn(self, state, action) -> float:
        # x = state[0] + action[0] - 1
        x = state[0]
        return 1 - ((x - 8) ** 2)

    def render(self, mode='human'):
        pass


class OneDTargetDiscDet(OneDTargetDisc):
    def __init__(self):
        super(OneDTargetDiscDet, self).__init__()
        self.n = 0

    def reset(self):
        self.st = np.array([self.n % 11], dtype=int)
        self.timesteps = 0.0
        self.n += 1
        return self._get_obs()

    def step(self, action: np.ndarray):
        ns, r, done, info = super(OneDTargetDiscDet, self).step(action)
        if done:
            if ns < 0:
                self.st += 1
            if ns > 10:
                self.st -= 1
        return self.st, r, None, info
