import os
import gym
import pickle
import numpy as np

from typing import Optional, Tuple
from copy import deepcopy

from algos.tabular.policy.policy import TabularPolicy, TabularSoftPolicy
from imitation.util import logger
from imitation.data import rollout
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv


class QLearning:
    def __init__(
            self,
            env,
            gamma: float = 0.8,
            epsilon: float = 0.4,
            alpha: float = 0.5,
            device: str = 'cpu',
            **kwargs,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.device = device
        self.env = env
        self.num_envs = 1
        if isinstance(self.env, VecEnv):
            self.num_envs = self.env.num_envs
        self._setup_model()

    def _setup_model(self):
        self.num_timesteps = 0
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        assert isinstance(self.observation_space, gym.spaces.MultiDiscrete)\
               and isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.policy = TabularPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            epsilon=self.epsilon,
            alpha=self.alpha,
            device=self.device,
        )

    def train(
            self,
            ob: np.ndarray,
            action: np.ndarray,
            next_ob: np.ndarray,
            reward: np.ndarray,
    ) -> None:
        ob_idx = self.policy.obs_to_idx(ob)
        act_idx = self.policy.act_to_idx(action)
        nob_idx = self.policy.obs_to_idx(next_ob)
        self.policy.q_table[ob_idx, act_idx] += \
            self.alpha * (reward + self.gamma * np.max(self.policy.q_table[nob_idx, :], axis=-1)
                          - self.policy.q_table[ob_idx, act_idx])

    def learn(self, total_timesteps, reset_num_timesteps=True, **kwargs) -> None:
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.policy.reset()
        else:
            total_timesteps += self.num_timesteps
        while self.num_timesteps < total_timesteps:
            prev_q_table = deepcopy(self.policy.q_table)
            sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=self.num_envs)
            if not isinstance(self.env, VecEnv):
                self.env = DummyVecEnv([lambda: self.env])
            trans = rollout.flatten_trajectories_with_rew(
                rollout.generate_trajectories(self.policy, self.env, sample_until, deterministic_policy=False)
            )
            self.train(trans.obs, trans.acts, trans.next_obs, trans.rews)
            self.policy.policy_table = self.policy.arg_max(self.policy.q_table)
            error = np.max(np.abs(self.policy.q_table - prev_q_table))
            if self.num_timesteps % 100 == 0:
                logger.record("num_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.record("Action-Value Error", error)
                logger.dump(self.num_timesteps)
            if error < 1e-5 and self.num_timesteps > total_timesteps / 2:
                logger.record("Action-Value Error", error)
                logger.dump(self.num_timesteps)
                break
            self.num_timesteps += self.num_envs

    def reset(self, env: None):
        if env is None:
            env = self.env
        self.set_env(env)
        self._setup_model()

    def get_env(self):
        return self.env

    def set_env(self, env):
        self.env = env
        self.num_envs = 1
        if isinstance(self.env, VecEnv):
            self.num_envs = self.env.num_envs

    def get_vec_normalize_env(self):
        return None

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.policy.predict(observation, state, mask, deterministic)

    @classmethod
    def load(
        cls,
        path,
        env,
        device,
    ):
        pass

    def save(self, log_dir):
        state = self.__dict__.copy()
        del state['env']
        self.__dict__.update(state)
        self.env = None
        with open(log_dir + ".tmp", "wb") as f:
            pickle.dump(self, f)
        os.replace(log_dir + ".tmp", log_dir + ".pkl")


class SoftQLearning(QLearning):
    def _setup_model(self):
        self.num_timesteps = 0
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        assert isinstance(self.observation_space, gym.spaces.MultiDiscrete) \
               and isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.policy = TabularSoftPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            epsilon=self.epsilon,
            alpha=self.alpha,
            device=self.device,
        )

    def train(
            self,
            ob: np.ndarray,
            action: np.ndarray,
            next_ob: np.ndarray,
            reward: np.ndarray,
    ) -> None:
        ob_idx = self.policy.obs_to_idx(ob)
        act_idx = self.policy.act_to_idx(action)
        nob_idx = self.policy.obs_to_idx(next_ob)
        self.policy.q_table[ob_idx, act_idx] += \
            self.alpha * (reward - self.policy.q_table[ob_idx, act_idx]
                          + self.gamma * self.logsumexp(self.policy.q_table[nob_idx, :]))

    def logsumexp(self, x):
        return np.max(x, axis=-1) + np.log(np.exp(x - np.max(x, axis=-1)[:, np.newaxis]).sum(axis=-1))
