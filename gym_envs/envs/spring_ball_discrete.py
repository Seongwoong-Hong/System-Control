import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import seeding
from matplotlib import cm
from typing import List
from scipy.sparse import csc_matrix


class SpringBallDisc(gym.Env):
    def __init__(self):
        self.map_size = 0.4
        self.dt = 0.05
        self.st = None
        self.mass = 1
        self.l0 = 0.1
        self.k = 1
        self.viewer = None
        self.target = np.array([0.1])

        self.num_cells = [100, 100]
        self.num_actions = [20]
        self.obs_low = np.array([-self.map_size, -2.0])
        self.obs_high = np.array([self.map_size, 2.0])
        self.acts_low = np.array([-10.0])
        self.acts_high = np.array([10.0])
        self.obs_list = []
        for high, low, n in zip(self.obs_high, self.obs_low, self.num_cells):
            self.obs_list.append(np.linspace(low, high, n + 1))
        self.acts_list = []
        for high, low, n in zip(self.acts_high, self.acts_low, self.num_actions):
            self.acts_list.append(np.linspace(low, high, n + 1))

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Box(low=self.acts_low, high=self.acts_high)
        self.seed()

    @property
    def disc_states(self):
        return [(os[1:] + os[:-1]) / 2 for os in self.obs_list]

    @property
    def disc_actions(self):
        return [(ts[1:] + ts[:-1]) / 2 for ts in self.acts_list]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random()
        return [seed]

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        assert action in self.action_space, f"{action} is Out of action space"
        self.last_a = action
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        r = self.get_reward(self.st, action)
        self.st = self._get_next_state(self.st, action)
        return self.st, r, None, info

    def reset(self):
        high = np.array([*self.obs_high[0]])
        low = np.array([*self.obs_low[0]])
        self.st = self.np_random.uniform(low=low, high=high)
        return self._get_obs()

    def set_state(self, st):
        assert st in self.observation_space
        self.st = st

    def get_reward(self, state, action):
        x, dx = np.split(state, 2, axis=-1)
        return - (x ** 2 - 2 * self.target * x + 0.1 * dx ** 2 + 1e-2 * action ** 2)
        # return - (x ** 2 - 2 * self.target * x)

    def _get_next_state(self, state, action):
        x, dx = np.split(state, 2, axis=-1)
        ddx = - (self.k / self.mass) * x + action / self.mass + self.k * self.l0 / self.mass
        next_state = state + np.append(dx, ddx, axis=-1) * self.dt
        return np.clip(next_state, a_min=self.obs_low, a_max=self.obs_high)

    def _get_obs(self):
        return self.st

    def get_num_cells(self):
        return self.num_cells

    def get_vectorized(self):
        s_vec = np.stack(np.meshgrid(*self.disc_states,
                                     indexing='ij'),
                         -1).reshape(-1, 2)
        a_vec = np.stack(np.meshgrid(*self.disc_actions,
                                     indexing='ij'),
                         -1).reshape(-1, 1)

        return s_vec, a_vec

    def get_init_vector(self):
        return self.get_vectorized()

    def get_idx_from_obs(self, obs: np.ndarray):
        if len(obs.shape) == 1:
            obs = obs[None, :]
        assert (np.max(obs, axis=0) <= self.obs_high + 1e-6).all() or (np.min(obs, axis=0) >= self.obs_low - 1e-6).all()
        dims = self.get_num_cells()
        idx = []
        for i, whole_candi in enumerate(self.obs_list):
            idx.append((obs[:, [i]] - whole_candi[:-1] >= 0).sum(axis=-1) - 1)
        tot_idx = np.ravel_multi_index(np.array(idx), dims, order='C')
        return tot_idx.flatten()

    def get_obs_from_idx(self, idx: np.ndarray):
        assert len(idx.shape) == 1
        s_vec = np.stack(np.meshgrid(*self.disc_states,
                                     indexing='ij'),
                         -1).reshape(-1, 2)
        return s_vec[idx]

    def get_idx_from_acts(self, acts: np.ndarray):
        if len(acts.shape) == 1:
            acts = acts[None, :]
        assert (np.max(acts, axis=0) <= self.acts_high + 1e-6).all() or (np.min(acts, axis=0) >= self.acts_low - 1e-6).all()
        dims = np.array(self.num_actions)
        idx = []
        for i, whole_candi in enumerate(self.acts_list):
            idx.append((acts[:, [i]] - whole_candi[:-1] >= 0).sum(axis=-1) - 1)
        tot_idx = np.ravel_multi_index(np.array(idx), dims, order='C')

        return tot_idx.flatten()

    def get_acts_from_idx(self, idx: np.ndarray):
        assert len(idx.shape) == 1
        a_vec = np.stack(np.meshgrid(*self.disc_actions,
                                     indexing='ij'),
                         -1).reshape(-1, 1)
        return a_vec[idx]

    def get_trans_mat(self):
        s_vec, a_vec = self.get_vectorized()
        P = []
        for a in a_vec:
            next_s_vec = self._get_next_state(s_vec, a)
            tot_idx = self.get_idx_from_obs(next_s_vec)
            P.append(csc_matrix((np.ones(len(s_vec)), (tot_idx, np.arange(len(s_vec)))),
                                shape=[len(s_vec), len(s_vec)]))
        return np.stack(P)

    def get_reward_mat(self):
        s_vec, a_vec = self.get_vectorized()
        R = []
        for a in a_vec:
            R.append(self.get_reward(s_vec, a).flatten())
        return np.stack(R)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(600, 100)
            self.viewer.set_bounds(-1.2 * self.map_size, 1.2 * self.map_size, -0.2 * self.map_size, 0.2 * self.map_size)

            # for x in range(len(self.disc_states) + 1):
            #     vertical = rendering.Line(start=(x, 0.), end=(x, self.map_size))
            #     horizontal = rendering.Line(start=(0., x), end=(self.map_size, x))
            #     self.viewer.add_geom(vertical)
            #     self.viewer.add_geom(horizontal)

            goal = rendering.make_circle(0.05)
            goal.set_color(0.3, 0.8, 0.3)
            goal_transform = rendering.Transform()
            goal.add_attr(goal_transform)
            goal_transform.set_translation(self.target[0], 0.0)
            self.viewer.add_geom(goal)

            agent = rendering.make_circle(0.05)
            agent.set_color(0.3, 0.3, 0.8)
            self.agent_transform = rendering.Transform()
            agent.add_attr(self.agent_transform)
            self.viewer.add_geom(agent)

        self.agent_transform.set_translation(self.st[0], 0.0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")


class SpringBallDiscDet(SpringBallDisc):
    def __init__(self, init_states=None):
        super().__init__()
        self.idx = 0
        if init_states is None:
            s_vec, _ = self.get_vectorized()
            self.init_states = s_vec[0:len(s_vec):len(s_vec)//15]
        else:
            self.init_states = np.array(init_states)

    def reset(self):
        self.st = self.init_states[self.idx]
        self.idx = (self.idx + 1) % len(self.init_states)
        return self._get_obs()

    def get_init_vector(self):
        s_vec = self.init_states.copy()
        _, a_vec = self.get_vectorized()
        return s_vec, a_vec
