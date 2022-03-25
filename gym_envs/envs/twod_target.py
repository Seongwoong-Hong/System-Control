import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List


class TwoDTarget(gym.Env):
    def __init__(self, map_size=1):
        self.map_size = map_size
        self.dt = 0.01
        self.st = None
        self.viewer = None
        self.timesteps = 0
        self.target = np.array([self.map_size / 3, self.map_size / 3])
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]),
                                                high=np.array([self.map_size, self.map_size]))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.action_coeff = 5

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        r = self.get_reward(self.st, action)
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        self.st = self._get_next_state(self.st, action * self.action_coeff)
        self.timesteps += 1
        return self._get_obs(), r, None, info

    def _get_obs(self):
        return self.st
        # return np.append(self.st, self.timesteps)

    def _get_next_state(self, state, action):
        state = state + action * self.dt
        return np.clip(state, a_max=self.map_size, a_min=0.0)

    def set_state(self, state):
        assert state in self.observation_space
        self.st = state
        self.timesteps = 0

    def reset(self):
        self.st = self.observation_space.sample()
        self.timesteps = 0
        return self._get_obs()

    def get_reward(self, state, action) -> float:
        return 1 - (((state - self.target) ** 2).sum() + 1e-2 * (action ** 2).sum())

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, self.map_size, 0, self.map_size)

        bot = self.viewer.draw_circle(self.map_size / 50)
        bot.set_color(0, 0, 0)
        bot_transform = rendering.Transform(translation=(self.st[0], self.st[1]))
        bot.add_attr(bot_transform)

        target = self.viewer.draw_circle(self.map_size / 50)
        target.set_color(0.8, 0.3, 0.3)
        target_transform = rendering.Transform(translation=(self.target[0], self.target[1]))
        target.add_attr(target_transform)

        return self.viewer.render(return_rgb_array=mode=='rgb_array')


class TwoDTargetDet(TwoDTarget):
    def __init__(self, map_size=1, init_states=None):
        super(TwoDTargetDet, self).__init__(map_size=map_size)
        if init_states is None:
            self.init_states = np.array([[-map_size / 3, map_size / 3],
                                        [-map_size / 3, -map_size / 3],
                                        [map_size / 3, -map_size / 3]])
        else:
            self.init_states = init_states
        self.n = 0

    def reset(self):
        self.st = self.init_states[self.n % len(self.init_states)]
        self.timesteps = 0
        self.n += 1
        return self._get_obs()
