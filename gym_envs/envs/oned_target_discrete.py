import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List


class OneDTargetDisc(gym.Env):
    def __init__(self):
        self.size = 0.5
        self.dt = 0.1
        self.st = None
        self.timesteps = 0.0
        self.observation_space = gym.spaces.Box(shape=(1,), low=-self.size, high=self.size, dtype=np.float64)
        self.action_space = gym.spaces.MultiDiscrete([3])

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        action = action.astype('float64') - 1.0
        r = self.reward_fn(self.st)
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        st = self.st + self.dt * action  # + 0.03 * np.random.random(self.st.shape)
        done = bool(
            self.st[0] < -self.size
            or self.st[0] > self.size
        )
        if not done:
            self.st = st
        self.timesteps += self.dt
        return self.st, r, None, info

    def _get_obs(self):
        return self.st

    def reset(self):
        self.st = np.round_(self.observation_space.sample(), 1)
        self.timesteps = 0.0
        return self._get_obs()

    def reward_fn(self, state) -> float:
        x = state[0]
        return - ((x - 0.4) ** 2)

    def render(self, mode='human'):
        pass


class OneDTargetDiscDet(OneDTargetDisc):
    def __init__(self):
        super(OneDTargetDiscDet, self).__init__()
        self.init_state = np.array([[-0.5], [-0.4], [-0.3], [-0.2], [-0.1], [0.0], [0.1], [0.2], [0.3], [0.4], [0.5]])
        self.n = 0

    def reset(self):
        self.st = self.init_state[self.n % len(self.init_state), :]
        self.timesteps = 0.0
        self.n += 1
        return self._get_obs()
