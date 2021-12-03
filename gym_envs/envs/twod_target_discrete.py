import gym
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List


class TwoDTargetDisc(gym.Env):
    def __init__(self, map_size=10):
        self.map_size = map_size
        self.act_size = 3
        self.dt = 0.1
        self.st = None
        self.timesteps = 0
        self.observation_space = gym.spaces.MultiDiscrete([self.map_size, self.map_size])
        self.action_space = gym.spaces.MultiDiscrete([self.act_size, self.act_size])

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        r = self.reward_fn(self.st, action)
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        self.st += action - 1
        if self.st[0] >= self.map_size:
            self.st[0] -= 1
        elif self.st[0] < 0:
            self.st[0] += 1
        if self.st[1] >= self.map_size:
            self.st[1] -= 1
        elif self.st[1] < 0:
            self.st[1] += 1
        self.timesteps += 1
        return self.st, r, None, info

    def set_state(self, state):
        self.st = state

    def _get_obs(self):
        return self.st

    def reset(self):
        self.st = self.observation_space.sample()
        self.timesteps = 0
        return self._get_obs()

    def reward_fn(self, state, action) -> float:
        x, y = state[0], state[1]
        return - ((x - 2) ** 2 + (y - 2) ** 2)

    def draw(self, trajs: List[np.ndarray] = None):
        if trajs is None:
            trajs = []
        d1, d2 = np.meshgrid(np.linspace(0, self.map_size, 100), np.linspace(0, self.map_size, 100))
        r = self.reward_fn([d1, d2], [1, 1])
        fig = plt.figure(figsize=[4, 4], dpi=300.0)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(d1, d2, r, rstride=1, cstride=1, cmap=cm.rainbow)
        # ax.pcolor(d1, d2, r, cmap=cm.rainbow)
        ax.set_xlabel("x", labelpad=15.0, fontsize=28)
        ax.set_ylabel("y", labelpad=15.0, fontsize=28)
        ax.set_title("Reward", fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # for traj in trajs:
        #     obs = []
        #     for ob_idx in traj[:, 0]:
        #         obs.append(self.idx_to_ob(ob_idx))
        #     ax.plot3D(np.array(obs)[:, 0], np.array(obs)[:, 1], traj[:, 1], color='k')
        ax.view_init(elev=90, azim=0)
        fig.show()

    def render(self, mode='human'):
        pass


class TwoDTargetDiscDet(TwoDTargetDisc):
    def __init__(self, map_size=10):
        super(TwoDTargetDiscDet, self).__init__(map_size=map_size)
        # x, y = np.meshgrid(range(0, map_size, 2), range(0, map_size, 2))
        x, y = np.meshgrid(range(map_size), range(map_size))
        self.init_state = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)
        self.n = 0

    def reset(self):
        self.st = deepcopy(self.init_state[self.n % len(self.init_state), :])
        self.timesteps = 0
        self.n += 1
        return self._get_obs()
