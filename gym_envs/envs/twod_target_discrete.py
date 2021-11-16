import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List


class TwoDTargetDisc(gym.Env):
    def __init__(self):
        self.map_size = 10
        self.act_size = 3
        self.dt = 0.1
        self.st = None
        self.timesteps = 0.0
        self.observation_space = gym.spaces.MultiDiscrete([self.map_size * self.map_size])
        self.action_space = gym.spaces.MultiDiscrete([self.act_size * self.act_size])

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        st = self.idx_to_ob(self.st)
        act = self.idx_to_act(action)
        r = self.reward_fn(st, act)
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        st += act - 1
        done = bool(
            st[0] < 0
            or st[0] >= self.map_size
            or st[1] < 0
            or st[1] >= self.map_size
        )
        if done:
            r -= 1000
            info['done'] = done
            if st[0] >= self.map_size:
                st[0] -= 1
            elif st[0] < 0:
                st[0] += 1
            if st[1] >= self.map_size:
                st[1] -= 1
            elif st[1] < 0:
                st[1] += 1
        self.timesteps += self.dt
        self.st = self.ob_to_idx(st)
        return self.st, r, done, info

    def set_state(self, state):
        self.st = state

    def _get_obs(self):
        return self.st

    def reset(self):
        self.st = self.observation_space.sample()
        self.timesteps = 0.0
        return self._get_obs()

    def idx_to_ob(self, ob_idx):
        return np.array([ob_idx.item() % self.map_size, ob_idx.item() / self.map_size], dtype=int)

    def idx_to_act(self, act_idx):
        return np.array([act_idx.item() % self.act_size, act_idx.item() / self.act_size], dtype=int)

    def ob_to_idx(self, ob):
        return np.array([ob[0] + 10 * ob[1]], dtype=int)

    def reward_fn(self, state, action) -> float:
        x, y = state[0] + action[0] - 1, state[1] + action[1] - 1
        return - ((x - 6) ** 2 + (y - 8) ** 2)

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
        for traj in trajs:
            obs = []
            for ob_idx in traj[:, 0]:
                obs.append(self.idx_to_ob(ob_idx))
            ax.plot3D(np.array(obs)[:, 0], np.array(obs)[:, 1], traj[:, 1], color='k')
        ax.view_init(elev=90, azim=0)
        fig.show()

    def render(self, mode='human'):
        pass


class TwoDTargetDiscDet(TwoDTargetDisc):
    def __init__(self):
        super(TwoDTargetDiscDet, self).__init__()
        self.init_state = range(0, 100, 5)
        self.n = 0

    def step(self, action: np.ndarray):
        next_ob, r, done, info = super().step(action)
        return next_ob, r, None, info

    def reset(self):
        self.st = np.array([self.init_state[self.n % len(self.init_state)]], dtype=int)
        self.timesteps = 0.0
        self.n += 1
        return self._get_obs()
