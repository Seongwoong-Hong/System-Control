import gym
import numpy as np
from copy import deepcopy
from gym.utils import seeding
from scipy.sparse import csc_matrix


class OneDTargetDisc(gym.Env):
    def __init__(self, map_size=50):
        self.dt = 0.1
        self.timesteps = 0
        self.map_size = map_size
        self.action_size = 9

        self.st = None

        self.observation_space = gym.spaces.MultiDiscrete([self.map_size])
        self.action_space = gym.spaces.MultiDiscrete([self.action_size])

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        r = self.get_reward(self.st, action)
        self.st = self._get_next_state(self.st, action)
        self.timesteps += 1
        return self.st, r, None, info

    def _get_obs(self):
        return self.st

    def get_torque(self, action):
        return action.T

    def _get_next_state(self, state, action):
        state = state + action - self.action_size // 2
        llim_idx = state < 0
        hlim_idx = state >= self.map_size
        state[llim_idx] = 0
        state[hlim_idx] = self.map_size - 1
        return state

    def set_state(self, obs: np.ndarray):
        assert self.map_size > obs.item() >= 0
        self.st = obs

    def reset(self):
        self.set_state(self.observation_space.sample())
        self.timesteps = 0
        return self._get_obs()

    def get_reward(self, state, action):
        return - ((state - 10) ** 2) - (action - self.action_size // 2) ** 2

    def get_vectorized(self):
        s_vec = np.array(range(self.map_size)).reshape(-1, 1)
        a_vec = np.array(range(self.action_size)).reshape(-1, 1)
        return s_vec, a_vec

    def get_init_vector(self):
        return self.get_vectorized()

    def get_idx_from_obs(self, obs: np.ndarray):
        return obs.flatten()

    def get_obs_from_idx(self, idx: np.ndarray):
        return idx.reshape(-1, 1)

    def get_act_from_idx(self, idx: np.ndarray):
        return idx.reshape(-1, 1)

    def get_idx_from_act(self, act: np.ndarray):
        return act.flatten()

    def get_trans_mat(self):
        # Transition Matrix shape: (|A|, |Next S|, |S|)
        s_vec, a_vec = self.get_vectorized()
        P = []
        for a in a_vec:
            next_s_vec = self._get_next_state(s_vec, a)
            tot_idx = self.get_idx_from_obs(next_s_vec)
            P.append(csc_matrix((np.ones(self.map_size), (tot_idx, np.arange(self.map_size))),
                                shape=[self.map_size, self.map_size]))
        return np.stack(P)

    def get_reward_mat(self):
        s_vec, a_vec = self.get_vectorized()
        R = []
        for a in a_vec:
            R.append(self.get_reward(s_vec, a).flatten())
        return np.stack(R)

    def render(self, mode='human'):
        pass


class OneDTargetDiscDet(OneDTargetDisc):
    def __init__(self, map_size=50, init_states=None):
        super(OneDTargetDiscDet, self).__init__(map_size=map_size)
        if init_states is None:
            self.init_states, _ = self.get_vectorized()
            self.init_states = self.init_states[0:len(self.init_states):3]
        else:
            self.init_states = self.get_obs_from_idx(self.get_idx_from_obs(np.array(init_states)))
        self.n = 0

    def reset(self):
        self.st = np.array([self.n % self.map_size], dtype=int)
        self.timesteps = 0
        self.n += 1
        return self._get_obs()

    def get_init_vector(self):
        s_vec = deepcopy(self.init_states)
        a_vec = np.arange(self.action_size).reshape(-1, 1)
        return s_vec, a_vec
