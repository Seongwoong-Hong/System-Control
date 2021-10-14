import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List


class TwoDWorld(gym.Env):
    def __init__(self):
        self.height = 5.0
        self.width = 5.0
        self.dt = 0.1
        self.st = None
        self.observation_space = gym.spaces.Box(low=np.array([-self.width, -self.height]),
                                                high=np.array([self.width, self.height]))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        r = self.reward_fn(self.st)
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        self.st = self.st + self.dt * action
        return self.st, r, None, info

    def _get_obs(self):
        return self.st

    def reset(self):
        self.st = np.random.uniform(low=[-self.width, -self.height], high=[self.width, self.height])
        return self._get_obs()

    def reward_fn(self, state) -> float:
        x, y = state[0], state[1]
        return np.exp(-0.5 * (x ** 2 + y ** 2))\
            - np.exp(-0.5 * ((x - self.width/2) ** 2 + (y - self.height/2) ** 2))\
            - np.exp(-0.5 * ((x + self.width/2) ** 2 + (y - self.height/2) ** 2))\
            - np.exp(-0.5 * ((x - self.width/2) ** 2 + (y + self.height/2) ** 2))\
            - np.exp(-0.5 * ((x + self.width/2) ** 2 + (y + self.height/2) ** 2))

    def draw(self, trajs: List[np.ndarray] = None):
        if trajs is None:
            trajs = []
        d1, d2 = np.meshgrid(np.linspace(-self.width, self.width, 100), np.linspace(-self.height, self.height, 100))
        r = self.reward_fn([d1, d2])
        fig = plt.figure(figsize=[4, 4], dpi=300.0)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(d1, d2, r, rstride=1, cstride=1, cmap=cm.rainbow)
        # ax.pcolor(d1, d2, r, cmap=cm.rainbow)
        ax.set_xlabel("x", labelpad=15.0, fontsize=28)
        ax.set_ylabel("y", labelpad=15.0, fontsize=28)
        ax.set_title("Reward", fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=20)
        for traj in trajs:
            ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], color='k')
        ax.view_init(elev=90, azim=0)
        fig.show()

    def render(self, mode='human'):
        pass


class TwoDWorldDet(TwoDWorld):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.init_state = 5 * np.array(([-0.5, 0.15],
                                       [-0.4, 0.15],
                                       [-0.3, 0.15],
                                       [-0.2, 0.15],
                                       [-0.1, 0.15],
                                       [0.0, 0.15],
                                       [0.1, 0.15],
                                       [0.2, 0.15],
                                       [0.3, 0.15],
                                       [0.4, 0.15],
                                       [0.5, 0.15],
                                       [-0.5, -0.15],
                                       [-0.4, -0.15],
                                       [-0.3, -0.15],
                                       [-0.2, -0.15],
                                       [-0.1, -0.15],
                                       [0.0, -0.15],
                                       [0.1, -0.15],
                                       [0.2, -0.15],
                                       [0.3, -0.15],
                                       [0.4, -0.15],
                                       [0.5, -0.15]))

    def reset(self):
        # idx = np.random.randint(len(self.init_state))
        self.st = self.init_state[self.i, :]
        self.i = (self.i + 1) % len(self.init_state)
        return self._get_obs()
