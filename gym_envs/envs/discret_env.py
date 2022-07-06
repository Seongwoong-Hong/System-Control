import gym
import numpy as np
from gym.utils import seeding


class BaseDiscEnv(gym.Env):
    def __init__(self):
        super(BaseDiscEnv, self).__init__()
        self.obs_list = None
        self.acts_list = None
        self.obs_high = None
        self.obs_low = None
        self.max_torques = None
        self.min_torques = None
        self.num_cells = None
        self.num_actions = None
        self.np_random = None
        self.seed()

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random()
        return [seed]

    @property
    def disc_states(self):
        return [(os[1:] + os[:-1]) / 2 for os in self.obs_list]

    @property
    def disc_actions(self):
        return [(ts[1:] + ts[:-1]) / 2 for ts in self.acts_list]

    def get_vectorized(self):
        s_vec = np.stack(np.meshgrid(*self.disc_states,
                                     indexing='ij'),
                         -1).reshape(-1, len(self.disc_states))
        a_vec = np.stack(np.meshgrid(*self.disc_actions,
                                     indexing='ij'),
                         -1).reshape(-1, len(self.disc_actions))

        return s_vec, a_vec

    def get_idx_from_obs(self, obs: np.ndarray):
        if len(obs.shape) == 1:
            obs = obs[None, :]
        assert (np.max(obs, axis=0) <= self.obs_high + 1e-6).all() or (np.min(obs, axis=0) >= self.obs_low - 1e-6).all()
        dims = self.num_cells
        idx = []
        for i, whole_candi in enumerate(self.obs_list):
            idx.append((obs[:, [i]] - whole_candi[:-1] >= 0).sum(axis=-1) - 1)
        tot_idx = np.ravel_multi_index(np.array(idx), dims, order='C')
        return tot_idx.flatten()

    def get_obs_from_idx(self, idx: np.ndarray):
        assert len(idx.shape) == 1
        s_vec = np.stack(np.meshgrid(*self.disc_states,
                                     indexing='ij'),
                         -1).reshape(-1, len(self.disc_states))
        return s_vec[idx]

    def get_acts_from_idx(self, idx: np.ndarray):
        assert len(idx.shape) == 1
        a_vec = np.stack(np.meshgrid(*self.disc_actions,
                                     indexing='ij'),
                         -1).reshape(-1, len(self.disc_actions))
        return a_vec[idx]

    def get_idx_from_acts(self, acts: np.ndarray):
        if len(acts.shape) == 1:
            acts = acts[None, :]
        assert (np.max(acts, axis=0) <= self.max_torques + 1e-6).all()
        dims = np.array(self.num_actions)
        idx = []
        for i, whole_candi in enumerate(self.acts_list):
            idx.append((acts[:, [i]] - whole_candi[:-1] >= 0).sum(axis=-1) - 1)
        tot_idx = np.ravel_multi_index(np.array(idx), dims, order='C')
        return tot_idx.flatten()

    def get_trans_mat(self):
        raise NotImplementedError