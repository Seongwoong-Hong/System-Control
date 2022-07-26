import gym
import numpy as np
from gym.utils import seeding


class DiscretizationInfo:
    def __init__(self):
        self.high = None
        self.low = None
        self.num_cells = None

    def set_info(self, *args, **kwargs):
        # 전체 discrete information 을 업데이트하는 함수
        raise NotImplementedError

    def get_vectorized(self):
        # 전체 discrete information 을 출력하는 함수
        raise NotImplementedError

    def get_info_from_idx(self, idx:np.ndarray):
        # [N,] shape 을 가진 index array 가 입력으로 들어오면 해당 index 에 대응되는 information 을 출력하는 함수
        raise NotImplementedError

    def get_idx_from_info(self, info:np.ndarray):
        # [N, x] shape 을 가진 information array 가 입력으로 들어오면 해당 information 에 대응되는 index 를 출력하는 함수
        raise NotImplementedError


class BaseDiscEnv(gym.Env):
    def __init__(self, obs_info: DiscretizationInfo, acts_info: DiscretizationInfo):
        super(BaseDiscEnv, self).__init__()
        self.np_random = None
        self.obs_info = obs_info
        self.acts_info = acts_info
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

    def get_vectorized(self):
        s_vec = self.obs_info.get_vectorized()
        a_vec = self.acts_info.get_vectorized()
        return s_vec, a_vec

    def get_idx_from_obs(self, obs: np.ndarray):
        if len(obs.shape) == 1:
            obs = obs[None, :]
        return self.obs_info.get_idx_from_info(obs)

    def get_obs_from_idx(self, idx: np.ndarray):
        assert len(idx.shape) == 1
        return self.obs_info.get_info_from_idx(idx)

    def get_acts_from_idx(self, idx: np.ndarray):
        assert len(idx.shape) == 1
        return self.acts_info.get_info_from_idx(idx)

    def get_idx_from_acts(self, acts: np.ndarray):
        if len(acts.shape) == 1:
            acts = acts[None, :]
        return self.acts_info.get_idx_from_info(acts)

    def get_trans_mat(self):
        raise NotImplementedError