import os
import gym
import pickle
import numpy as np
from typing import Optional, Tuple
from copy import deepcopy
from algos.tabular.policy import *
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from imitation.util import logger


class Viter:
    def __init__(
            self,
            env,
            gamma: float = 0.9,
            epsilon: float = 0.4,
            alpha: float = 4,
            device: str = 'cpu',
            **kwargs,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.device = device
        self.env = env
        if not isinstance(env, VecEnv):
            self.env = DummyVecEnv([lambda: env])
        assert self.env.num_envs == 1, "Multiple environments are not available"
        self._setup_model()
        self.set_env_mats()

    def _setup_model(self):
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.num_timesteps = 0
        assert isinstance(self.observation_space, gym.spaces.MultiDiscrete) \
               and isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.policy = TabularPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            epsilon=self.epsilon,
            alpha=self.alpha,
            device=self.device,
        )

    def set_env_mats(self):
        self.transition_mat = np.zeros([self.policy.obs_size, self.policy.obs_size, self.policy.act_size])
        self.reward_mat = np.zeros([self.policy.obs_size, self.policy.act_size])
        self.done_mat = np.zeros([self.policy.obs_size, self.policy.act_size])
        for i in range(self.policy.obs_size):
            for j in range(self.policy.act_size):
                self.env.reset()
                states = self.policy.idx_to_obs(np.array([i]))
                actions = self.policy.idx_to_act(np.array([[j]]))
                self.env.env_method("set_state", states[0])
                ns, r, done, _ = self.env.step(actions)
                k = self.policy.obs_to_idx(ns).item()
                self.transition_mat[k, i, j] = 1
                self.reward_mat[i, j] = r[0]
                if done:
                    self.done_mat[i, j] = 1

    def train(self):
        self.policy.q_table = self.reward_mat + self.gamma * (1 - self.done_mat) * np.sum(
            self.transition_mat * np.max(self.policy.q_table, axis=-1)[:, None, None], axis=0)
        self.policy.v_table = np.max(self.policy.q_table, axis=-1)
        self.policy.policy_table = np.zeros(self.policy.policy_table.shape)
        self.policy.policy_table[range(self.policy.obs_size), self.policy.arg_max(self.policy.q_table).flatten()] = 1

    def learn(self, total_timesteps, reset_num_timesteps=True, **kwargs):
        min_timesteps = kwargs.pop("min_timesteps", 10)
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.policy.reset()
        else:
            total_timesteps += self.num_timesteps
        self.set_env_mats()
        while self.num_timesteps < total_timesteps:
            old_value = deepcopy(self.policy.v_table)
            self.train()
            error = np.max(np.abs(old_value - self.policy.v_table))
            if self.num_timesteps % 10 == 0:
                logger.record("num_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.record("Value Error", error)
                logger.dump(self.num_timesteps)
            if error < 1e-10 and self.num_timesteps >= min_timesteps:
                logger.record("num_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.record("Value Error", error)
                self.policy.policy_table = np.round(self.policy.policy_table, 8)
                break
            self.num_timesteps += 1

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.policy.predict(observation, state, mask, deterministic)

    def logsumexp(self, x, axis=0):
        assert len(x.shape) != 1
        return np.max(x, axis=axis) + np.log(np.sum(np.exp(x - np.max(x, axis=axis)[:, None]), axis=axis))

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
        assert self.env.num_envs == 1, "Multiple environments are not available"
        self.num_envs = 1

    def get_vec_normalize_env(self):
        return None

    def save(self, log_dir):
        state = self.__dict__.copy()
        del state['env']
        self.__dict__.update(state)
        self.env = None
        with open(log_dir + ".tmp", "wb") as f:
            pickle.dump(self, f)
        os.replace(log_dir + ".tmp", log_dir + ".pkl")


class SoftQiter(Viter):
    def train(self):
        self.policy.v_table = self.alpha * self.logsumexp(self.policy.q_table / self.alpha, axis=-1)
        self.policy.q_table = self.reward_mat + self.gamma * (1 - self.done_mat) * np.sum(
            self.transition_mat * self.policy.v_table[:, None, None], axis=0)
        self.policy.policy_table = np.exp((self.policy.q_table - self.policy.v_table[:, None]) / self.alpha)


class FiniteSoftQiter(Viter):
    def _setup_model(self):
        self.num_timesteps = 0
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.T = 200  # self.env.max_time
        assert isinstance(self.observation_space, gym.spaces.MultiDiscrete) \
               and isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.policy = FiniteTabularSoftPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            max_t=self.T
        )

    def train(self):
        self.policy.q_table[self.T - 1, :, :] = self.reward_mat
        self.policy.v_table[self.T - 1, :] = self.alpha * self.logsumexp(
            self.policy.q_table[self.T - 1, :, :] / self.alpha, axis=1)
        for t in reversed(range(self.T - 1)):
            self.policy.q_table[t, :, :] = self.reward_mat + self.gamma * np.sum(
                self.transition_mat * self.policy.v_table[t + 1, :][:, None, None], axis=0)
            self.policy.v_table[t, :] = self.alpha * self.logsumexp(self.policy.q_table[t, :, :] / self.alpha, axis=1)
        self.policy.policy_table = np.exp((self.policy.q_table - self.policy.v_table[:, :, None]) / self.alpha)
