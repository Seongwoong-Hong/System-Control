import os
import gym
import pickle
import numpy as np
from typing import Optional, Tuple
from copy import deepcopy
from algos.tabular.policy import TabularPolicy, TabularSoftPolicy
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

    def _setup_model(self, ):
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

    def train(self, p_mat, r_mat, d_mat):
        self.policy.q_table = r_mat + self.gamma * (1 - d_mat) * \
                              np.sum(p_mat * np.max(self.policy.q_table, axis=-1)[:, np.newaxis, np.newaxis], axis=0)
        self.policy.v_table = np.max(self.policy.q_table, axis=-1)

    def learn(self, total_timesteps, reset_num_timesteps=True, **kwargs):
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.policy.reset()
        else:
            total_timesteps += self.num_timesteps
        p_mat = np.zeros([self.policy.obs_size, self.policy.obs_size, self.policy.act_size])
        r_mat = np.zeros([self.policy.obs_size, self.policy.act_size])
        d_mat = np.zeros([self.policy.obs_size, self.policy.act_size])
        for i in range(self.policy.obs_size):
            for j in range(self.policy.act_size):
                self.env.reset()
                states = self.policy.idx_to_obs(np.array([i]))
                actions = self.policy.idx_to_act(np.array([[j]]))
                self.env.env_method("set_state", states[0])
                ns, r, done, _ = self.env.step(actions)
                k = self.policy.obs_to_idx(ns).item()
                p_mat[k, i, j] = 1
                r_mat[i, j] = r[0]
                if done:
                    d_mat[i, j] = 1
        while self.num_timesteps < total_timesteps:
            old_value = deepcopy(self.policy.v_table)
            self.train(p_mat, r_mat, d_mat)
            error = np.max(np.abs(old_value - self.policy.v_table))
            if self.num_timesteps % 10 == 0:
                logger.record("num_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.record("Value Error", error)
                logger.dump(self.num_timesteps)
            if error < 1e-8 and self.num_timesteps >= 100:
                logger.record("num_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.record("Value Error", error)
                self.policy.policy_table = self.policy.arg_max(self.policy.q_table)
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
    def _setup_model(self):
        self.num_timesteps = 0
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        assert isinstance(self.observation_space, gym.spaces.MultiDiscrete) \
               and isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.policy = TabularSoftPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
        )

    def train(self, p_mat, r_mat, d_mat):
        self.policy.q_table = r_mat + self.gamma * (1 - d_mat) * self.alpha * \
                              np.sum(p_mat * np.log(np.sum(np.exp(self.policy.q_table / self.alpha), axis=-1))[:,
                                             np.newaxis, np.newaxis], axis=0)
        self.policy.v_table = np.log(np.sum(np.exp(self.policy.q_table), axis=-1))
