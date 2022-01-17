import gym
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List
from scipy.sparse import csc_matrix


class TwoDTargetDisc(gym.Env):
    def __init__(self, map_size=50):
        self.map_size = map_size
        self.act_size = 7
        self.dt = 0.1
        self.st = None
        self.timesteps = 0
        self.observation_space = gym.spaces.MultiDiscrete([self.map_size, self.map_size])
        self.action_space = gym.spaces.MultiDiscrete([self.act_size, self.act_size])

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        r = self.get_reward(self.st, action)
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        self.st = self._get_next_state(self.st, action)
        self.timesteps += 1
        return self.st, r, None, info

    def _get_obs(self):
        return self.st

    def get_torque(self, action):
        return action.T

    def _get_next_state(self, state, action):
        state = state + action - self.act_size // 2
        return np.clip(state, a_max=self.map_size - 1, a_min=0)

    def set_state(self, state):
        assert state in self.observation_space
        self.st = state

    def reset(self):
        self.st = self.observation_space.sample()
        self.timesteps = 0
        return self._get_obs()

    def get_reward(self, state, action):
        target = np.array([10, 15])
        return - ((state - target) ** 2 + (action - self.act_size // 2) ** 2).sum(axis=-1, keepdims=True)

    def get_vectorized(self):
        s_vec = np.stack(np.meshgrid(range(self.map_size), range(self.map_size), indexing='ij'),
                         -1).reshape(-1, 2)
        a_vec = np.stack(np.meshgrid(range(self.act_size), range(self.act_size), indexing='ij'),
                         -1).reshape(-1, 2)
        return s_vec, a_vec

    def get_init_vector(self):
        return self.get_vectorized()

    def get_idx_from_obs(self, obs: np.ndarray):
        tot_idx = np.ravel_multi_index(obs.T, [self.map_size, self.map_size], order='C')
        return tot_idx.flatten()

    def get_obs_from_idx(self, idx: np.ndarray):
        s_vec, _ = self.get_vectorized()
        return s_vec[idx.flatten()]

    def get_idx_from_acts(self, act: np.ndarray):
        tot_idx = np.ravel_multi_index(act.T, [self.act_size, self.act_size], order='C')
        return tot_idx.flatten()

    def get_acts_from_idx(self, idx: np.ndarray):
        _, a_vec = self.get_vectorized()
        return a_vec[idx.flatten()]

    def get_trans_mat(self):
        s_vec, a_vec = self.get_vectorized()
        P = []
        for a in a_vec:
            next_s_vec = self._get_next_state(s_vec, a)
            tot_idx = self.get_idx_from_obs(next_s_vec)
            P.append(csc_matrix((np.ones(self.map_size ** 2), (tot_idx, np.arange(self.map_size ** 2))),
                                shape=[self.map_size ** 2, self.map_size ** 2]))
        return np.stack(P)

    def get_reward_mat(self):
        s_vec, a_vec = self.get_vectorized()
        R = []
        for a in a_vec:
            R.append(self.get_reward(s_vec, a).flatten())
        return np.stack(R)

    def draw(self, trajs: List[np.ndarray] = None):
        if trajs is None:
            trajs = []
        d1, d2 = np.meshgrid(np.linspace(0, self.map_size, 100), np.linspace(0, self.map_size, 100))
        r = self.get_reward(np.array([d1, d2]), np.array([self.act_size // 2, self.act_size // 2]))
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
    def __init__(self, map_size=50, init_states=None):
        super(TwoDTargetDiscDet, self).__init__(map_size=map_size)
        if init_states is None:
            self.init_states, _ = self.get_vectorized()
            self.init_states = self.init_states[0:len(self.init_states):3]
        else:
            self.init_states = self.get_obs_from_idx(self.get_idx_from_obs(np.array(init_states)))
        self.n = 0

    def reset(self):
        self.st = deepcopy(self.init_states[self.n % len(self.init_states), :])
        self.timesteps = 0
        self.n += 1
        return self._get_obs()

    def get_init_vector(self):
        s_vec = deepcopy(self.init_states)
        _, a_vec = self.get_vectorized()
        return s_vec, a_vec
