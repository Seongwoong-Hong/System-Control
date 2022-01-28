import os
import gym
import pickle
import numpy as np
import torch as th
from typing import Optional, Tuple, List
from copy import deepcopy
from algos.tabular.policy import *
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from imitation.util import logger


def backward_trans(P: List[th.tensor], v: th.tensor):
    """
    모든 A 에 대해 전환 행렬 P 를 고려할 때 이전 스텝의 v 계산
    P.shape = (|A|, |Next S|, |S|)
    A.shape = (|A|, |S|)
    post_v.shape = (|A|, |S|)
    """
    post_v = []
    for Pa in P:
        post_v.append(Pa.t().matmul(v))

    post_v = th.stack(post_v)

    return post_v


class Viter:
    def __init__(
            self,
            env,
            gamma: float = 0.9,
            epsilon: float = 0.4,
            alpha: float = 4,
            device: str = 'cpu',
            verbose: bool = True,
            **kwargs,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.device = device
        self.env = env
        self.verbose = verbose
        if not isinstance(env, VecEnv):
            self.env = DummyVecEnv([lambda: env])
        transition_mat = self.env.env_method('get_trans_mat')[0]
        self.transition_mat = []
        for csr in transition_mat:
            coo = csr.tocoo()
            self.transition_mat.append(th.sparse_coo_tensor(
                th.LongTensor(np.vstack((coo.row, coo.col))), th.FloatTensor(coo.data), th.Size(coo.shape),
            ).to(self.device))
        self._setup_model()
        self.set_reward_mats()

    def _setup_model(self):
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.num_timesteps = 0
        assert isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.policy = TabularPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            env=self.env.envs[0],
            epsilon=self.epsilon,
            alpha=self.alpha,
            device=self.device,
        )

    def set_reward_mats(self):
        self.reward_mat = self.env.env_method('get_reward_mat')[0]
        if isinstance(self.reward_mat, np.ndarray):
            self.reward_mat = th.from_numpy(self.reward_mat).float().to(self.device)
        if self.reward_mat.shape != self.policy.q_table.shape[-2:]:
            self.reward_mat = self.reward_mat.repeat(self.policy.act_size, 1)
        self.done_mat = th.zeros([self.policy.act_size, self.policy.obs_size]).float().to(self.device)

    def train(self):
        self.policy.v_table, max_idx = th.max(self.policy.q_table, dim=0)
        self.policy.q_table = self.reward_mat + self.gamma * (1 - self.done_mat) * \
                              backward_trans(self.transition_mat, self.policy.v_table)
        self.policy.policy_table = th.zeros([self.policy.act_size, self.policy.obs_size])
        self.policy.policy_table[max_idx, range(self.policy.obs_size)] = 1

    def learn(self, total_timesteps, reset_num_timesteps=True, **kwargs):
        min_timesteps = kwargs.pop("min_timesteps", 10)
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.policy.reset()
        else:
            total_timesteps += self.num_timesteps
        self.set_reward_mats()
        while True:
            old_value = deepcopy(self.policy.v_table)
            self.train()
            error = th.max(th.abs(old_value - self.policy.v_table)).item()
            if self.num_timesteps % 10 == 0 and self.verbose:
                logger.record("num_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.record("Value Error", error, exclude="tensorboard")
                logger.dump(self.num_timesteps)
            if self.num_timesteps >= total_timesteps or error < 1e-8:
                if self.verbose:
                    logger.record("num_timesteps", self.num_timesteps, exclude="tensorboard")
                    logger.record("Value Error", error, exclude="tensorboard")
                    logger.dump(self.num_timesteps)
                self.policy.policy_table = th.round(self.policy.policy_table * 1e8) * 1e-8
                break
            self.num_timesteps += 1

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        action, _ = self.policy.predict(observation, state, mask, deterministic)
        if not deterministic:
            eps = np.random.random()
            if eps < self.epsilon:
                action = self.action_space.sample()
        return action, None

    def reset(self, env: None):
        if env is None:
            env = self.env
        self.set_env(env)
        self._setup_model()

    def get_env(self):
        return self.env

    def set_env(self, env):
        self.env = env
        if not isinstance(env, VecEnv):
            self.env = DummyVecEnv([lambda: env])
        self.policy.env = self.env.envs[0]
        self.num_envs = 1
        transition_mat = self.env.env_method('get_trans_mat')[0]
        self.transition_mat = []
        for csr in transition_mat:
            coo = csr.tocoo()
            self.transition_mat.append(th.sparse_coo_tensor(
                th.LongTensor(np.vstack((coo.row, coo.col))), th.FloatTensor(coo.data), th.Size(coo.shape),
            ).to(self.device))

    def get_vec_normalize_env(self):
        return None

    def save(self, log_dir):
        env = deepcopy(self.env)
        policy_env = deepcopy(self.policy.env)
        self.env = None
        self.policy.env = None
        with open(log_dir + ".tmp", "wb") as f:
            pickle.dump(self, f)
        os.replace(log_dir + ".tmp", log_dir + ".pkl")
        self.env = env
        self.policy.env = policy_env


class SoftQiter(Viter):
    def train(self):
        self.policy.v_table = self.alpha * th.logsumexp(self.policy.q_table / self.alpha, dim=0)
        self.policy.q_table = self.reward_mat + self.gamma * (1 - self.done_mat) * \
                              backward_trans(self.transition_mat, self.policy.v_table)
        self.policy.policy_table = th.exp((self.policy.q_table - self.policy.v_table[None, :]) / self.alpha)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.policy.predict(observation, state, mask, deterministic)


class FiniteViter(Viter):
    def _setup_model(self):
        self.num_timesteps = 0
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        assert hasattr(self.env.get_attr("spec")[0], "max_episode_steps"), "Need to be specified the maximum timestep"
        self.max_t = self.env.get_attr("spec")[0].max_episode_steps
        assert isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.policy = FiniteTabularPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            env=self.env.envs[0],
            device=self.device,
            max_t=self.max_t,
        )

    def train(self):
        self.policy.q_table[-1] = self.reward_mat
        self.policy.v_table[-1], max_idx = th.max(self.policy.q_table[-1], dim=0)
        self.policy.policy_table[-1] = th.zeros([self.policy.act_size, self.policy.obs_size])
        self.policy.policy_table[-1, max_idx, range(self.policy.obs_size)] = 1
        for t in reversed(range(self.max_t - 1)):
            self.policy.q_table[t] = self.reward_mat + self.gamma * \
                                     backward_trans(self.transition_mat, self.policy.v_table[t + 1])
            self.policy.v_table[t], max_idx = th.max(self.policy.q_table[t], dim=0)
            self.policy.policy_table[t] = th.zeros([self.policy.act_size, self.policy.obs_size])
            self.policy.policy_table[t, max_idx, range(self.policy.obs_size)] = 1

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        obs_list, act_list, rew_list = [], [], []
        self.env.reset()
        self.env.env_method("set_state", observation.squeeze())
        for t in range(self.max_t):
            obs_list.append(observation.flatten())
            obs_idx = self.env.envs[0].get_idx_from_obs(observation)
            act_idx = self.policy.arg_max(self.policy.policy_table[t, :, obs_idx].T)
            act = self.env.envs[0].get_acts_from_idx(act_idx)
            if not deterministic:
                eps = np.random.random()
                if eps < self.epsilon:
                    act = self.action_space.sample()[None, :]
            observation, reward, _, _ = self.env.step(act)
            rew_list.append(reward.flatten())
            act_list.append(act.flatten())
        return np.array(obs_list), np.array(act_list), np.array(rew_list)


class FiniteSoftQiter(FiniteViter):
    def train(self):
        self.policy.q_table[-1] = self.reward_mat
        self.policy.v_table[-1] = self.alpha * th.logsumexp(self.policy.q_table[-1] / self.alpha, dim=0)
        for t in reversed(range(self.max_t - 1)):
            self.policy.q_table[t] = self.reward_mat + self.gamma * \
                                     backward_trans(self.transition_mat, self.policy.v_table[t + 1])
            self.policy.v_table[t] = self.alpha * th.logsumexp(self.policy.q_table[t] / self.alpha, dim=0)
        self.policy.policy_table = th.exp((self.policy.q_table - self.policy.v_table[:, None, :]) / self.alpha)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if not deterministic:
            choose_method = self.policy.choice_act
        else:
            choose_method = self.policy.arg_max
        obs_list, act_list, rew_list = [], [], []
        self.env.reset()
        self.env.env_method("set_state", observation.squeeze())
        for t in range(self.max_t):
            obs_list.append(observation.flatten())
            obs_idx = self.env.envs[0].get_idx_from_obs(observation)
            act_idx = choose_method(self.policy.policy_table[t, :, obs_idx].T)
            act = self.env.envs[0].get_acts_from_idx(act_idx)
            observation, reward, _, _ = self.env.step(act)
            rew_list.append(reward.flatten())
            act_list.append(act.flatten())
        return np.array(obs_list), np.array(act_list), np.array(rew_list)
