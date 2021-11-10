import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List


class TwoDTargetDisc(gym.Env):
    def __init__(self):
        self.height = 0.5
        self.width = 0.5
        self.dt = 0.1
        self.st = None
        self.timesteps = 0.0
        self.observation_space = gym.spaces.Box(low=np.array([-self.width, -self.height]),
                                                high=np.array([self.width, self.height]), dtype=np.float64)
        self.action_space = gym.spaces.MultiDiscrete([3, 3])

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        action = action.astype('float64') - 1.0
        r = self.reward_fn(self.st)
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        st = self.st + self.dt * action  # + 0.03 * np.random.random(self.st.shape)
        done = bool(
            self.st[0] < -self.width
            or self.st[0] > self.width
            or self.st[1] < -self.height
            or self.st[1] > self.height
        )
        if not done:
            self.st = st
        self.timesteps += self.dt
        return self.st, r, None, info

    def _get_obs(self):
        return self.st
        # return np.append(self.st, self.timesteps)

    def reset(self):
        self.st = np.round_(self.observation_space.sample(), 1)
        self.timesteps = 0.0
        return self._get_obs()

    def reward_fn(self, state) -> float:
        x, y = state[0], state[1]
        return - ((x - 2 * self.width/3) ** 2 + (y - 2 * self.height/3) ** 2)

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


class TwoDTargetDiscDet(TwoDTargetDisc):
    def __init__(self):
        super(TwoDTargetDiscDet, self).__init__()
        self.init_state = np.array([[-0.3, 0.3],
                                    [-0.3, -0.3],
                                    [0.3, -0.3]])
        self.n = 0

    def reset(self):
        self.st = self.init_state[self.n % len(self.init_state), :]
        self.timesteps = 0.0
        self.n += 1
        return self._get_obs()
