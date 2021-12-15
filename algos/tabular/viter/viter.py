import os
import gym
import pickle
import numpy as np
from typing import Optional, Tuple
from copy import deepcopy
from algos.tabular.policy import *
from algos.tabular.viter import backward_trans
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
        self._setup_model()
        self.set_env_mats()

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

    def set_env_mats(self):
        self.transition_mat = self.env.env_method('get_trans_mat')[0]
        self.reward_mat = self.env.env_method('get_reward_mat')[0]
        if self.reward_mat.shape != self.policy.q_table.shape[-2:]:
            self.reward_mat = np.repeat(self.reward_mat.reshape(1, -1), self.policy.act_size, axis=0)
        self.done_mat = np.zeros([self.policy.act_size, self.policy.obs_size])

    def train(self):
        self.policy.q_table = self.reward_mat + self.gamma * (1 - self.done_mat) * \
                              backward_trans(self.transition_mat, np.max(self.policy.q_table, axis=0))
        self.policy.v_table = np.max(self.policy.q_table, axis=0)
        state_policy = self.policy.q_table == self.policy.q_table.max(axis=0).reshape(1, -1)
        self.policy.policy_table = state_policy / np.sum(state_policy, axis=0).reshape(1, -1)

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
                logger.record("Value Error", error, exclude="tensorboard")
                logger.dump(self.num_timesteps)
            if error < 1e-10:
                logger.record("num_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.record("Value Error", error, exclude="tensorboard")
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
        return np.max(x, axis=axis) + np.log(np.sum(np.exp(x - np.max(x, axis=axis, keepdims=True)), axis=axis))

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

    def get_vec_normalize_env(self):
        return None

    def save(self, log_dir):
        state = self.__dict__.copy()
        del state['env']
        self.__dict__.update(state)
        self.env = None
        self.policy.env = None
        with open(log_dir + ".tmp", "wb") as f:
            pickle.dump(self, f)
        os.replace(log_dir + ".tmp", log_dir + ".pkl")


class SoftQiter(Viter):
    def train(self):
        self.policy.v_table = self.alpha * self.logsumexp(self.policy.q_table / self.alpha, axis=0)
        self.policy.q_table = self.reward_mat + self.gamma * (1 - self.done_mat) * \
                              backward_trans(self.transition_mat, self.policy.v_table)
        self.policy.policy_table = np.exp((self.policy.q_table - self.policy.v_table[None, :]) / self.alpha)


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
        self.policy.q_table[self.max_t - 1, :, :] = self.reward_mat
        self.policy.v_table[self.max_t - 1, :] = np.max(self.policy.q_table[self.max_t - 1, :, :], axis=0)
        policy_t = np.zeros([self.policy.obs_size, self.policy.act_size])
        policy_t[
            self.policy.arg_max(self.policy.q_table[self.max_t - 1, :, :]).flatten(), range(self.policy.obs_size)] = 1
        self.policy.policy_table[self.max_t - 1] = policy_t
        for t in reversed(range(self.max_t - 1)):
            self.policy.q_table[t, :, :] = self.reward_mat + self.gamma * np.sum(
                self.transition_mat * self.policy.v_table[t + 1, :][None, None, :], axis=-1)
            self.policy.v_table[t, :] = np.max(self.policy.q_table[t, :, :], axis=0)
            policy_t = np.zeros([self.policy.obs_size, self.policy.act_size])
            policy_t[self.policy.arg_max(self.policy.q_table[self.max_t - 1, :, :]).flatten(), range(
                self.policy.obs_size)] = 1
            self.policy.policy_table[t] = policy_t

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not deterministic:
            choose_method = self.policy.choice_act
        else:
            choose_method = self.policy.arg_max
        obs_list, act_list = [], []
        self.env.reset()
        self.env.env_method("set_state", observation[0])
        for t in range(self.max_t):
            obs_list.append(observation.flatten())
            obs_idx = self.env.envs[0].get_idx_from_obs(observation)
            act_idx = choose_method(self.policy.policy_table[t, :, obs_idx])
            act = self.env.envs[0].get_act_from_idx(act_idx)
            observation, _, _, _ = self.env.step(act)
            act_list.append(act.flatten())
        return np.array(obs_list), np.array(act_list)


class FiniteSoftQiter(FiniteViter):
    def train(self):
        self.policy.q_table[self.max_t - 1, :, :] = self.reward_mat
        self.policy.v_table[self.max_t - 1, :] = self.alpha * self.logsumexp(
            self.policy.q_table[self.max_t - 1, :, :] / self.alpha, axis=0)
        for t in reversed(range(self.max_t - 1)):
            self.policy.q_table[t, :, :] = self.reward_mat + self.gamma * \
                                           backward_trans(self.transition_mat, self.policy.v_table[t + 1, :])
            self.policy.v_table[t, :] = self.alpha * self.logsumexp(self.policy.q_table[t, :, :] / self.alpha, axis=0)
        self.policy.policy_table = np.exp((self.policy.q_table - self.policy.v_table[:, None, :]) / self.alpha)
