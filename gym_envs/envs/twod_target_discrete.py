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
        assert state in self.observation_space
        self.st = state

    def get_trans_mat(self):
        trans_mat = np.zeros([self.act_size ** 2, self.map_size ** 2, self.map_size ** 2])
        for j in range(self.map_size ** 2):
            for i in range(self.act_size ** 2):
                self.reset()
                self.set_state(np.array([j % self.map_size, j // self.map_size]))
                ns, _, _, _ = self.step(np.array([i % self.act_size, i // self.act_size]))
                k = ns[0] + ns[1] * self.map_size
                trans_mat[i, k, j] = 1
        return trans_mat

    def get_reward_vec(self):
        s_vec, a_vec = self.get_vectorized()
        return self.reward_fn(s_vec, a_vec)

    def get_vectorized(self):
        x, y = np.meshgrid(range(self.map_size), range(self.map_size))
        s_vec = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)
        a_vec = np.array(range(self.act_size ** 2)).reshape(-1, 1)
        return s_vec, a_vec

    def _get_obs(self):
        return self.st

    def reset(self):
        self.st = self.observation_space.sample()
        self.timesteps = 0
        return self._get_obs()

    def reward_fn(self, state, action) -> float:
        if state.ndim == 1:
            x, y = state
        else:
            x, y = np.split(state, 2, axis=-1)
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

    # TODO: Dimension이 1인 경우와 나누기
    def get_idx_from_obs(self, obs: np.ndarray) -> np.ndarray:
        obs_idx = obs[:, 0] + self.map_size * obs[:, 1]
        return obs_idx

    def get_obs_from_idx(self, idx: np.ndarray) -> np.ndarray:
        idx = idx.reshape(-1, 1)
        obs = np.append(idx % self.map_size, idx // self.map_size, axis=-1)
        return obs

    def get_idx_from_act(self, act: np.ndarray) -> np.ndarray:
        act_idx = act[:, 0] + self.action_space.nvec[0] * act[:, 1]
        return act_idx

    def get_act_from_idx(self, idx: np.ndarray) -> np.ndarray:
        idx = idx.reshape(-1, 1)
        act = np.append(idx % self.action_space.nvec[0], idx // self.action_space.nvec[0], axis=-1)
        return act


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
