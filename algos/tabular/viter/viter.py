import os
import gym
import pickle
import numpy as np
from typing import Optional, Tuple
from copy import deepcopy
from algos.tabular.viter.policy import TabularPolicy
from stable_baselines3.common.vec_env import VecEnv
from imitation.util import logger


class Viter:
    def __init__(
            self,
            env,
            gamma,
            epsilon,
            device,
            **kwargs,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self._setup_model(env)

    def _setup_model(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_timesteps = 0
        assert isinstance(self.observation_space, gym.spaces.MultiDiscrete) \
               and isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.policy = TabularPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            epsilon=self.epsilon,
        )

    def train(self):
        for i in range(self.observation_space.nvec[0]):
            candi = []
            for j in range(self.action_space.nvec[0]):
                self.env.reset()
                if isinstance(self.env, VecEnv):
                    self.env.env_method("set_state", np.array([i], dtype=int))
                    ns, r, done, _ = self.env.step(np.array([[j]], dtype=int))
                else:
                    self.env.set_state(np.array([i], dtype=int))
                    ns, r, done, _ = self.env.step(np.array([j], dtype=int))
                if not done:
                    candi.append(r + self.gamma * self.policy.v_table[ns])
                else:
                    candi.append(r)
            self.policy.v_table[i] = np.max(candi)

    def learn(self, total_timesteps, reset_num_timesteps=True, **kwargs):
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.policy.reset()
        else:
            total_timesteps += self.num_timesteps
        while self.num_timesteps < total_timesteps:
            old_value = deepcopy(self.policy.v_table)
            self.train()
            error = np.max(np.abs(old_value - self.policy.v_table))
            logger.record("num_timesteps", self.num_timesteps, exclude="tensorboard")
            logger.record("Value Error", error)
            logger.dump(self.num_timesteps)
            if error < 1e-8 and self.num_timesteps >= 100:
                for i in range(self.observation_space.nvec[0]):
                    candi = []
                    for j in range(self.action_space.nvec[0]):
                        self.env.reset()
                        if isinstance(self.env, VecEnv):
                            self.env.env_method("set_state", np.array([i], dtype=int))
                            ns, r, done, _ = self.env.step(np.array([[j]], dtype=int))
                        else:
                            self.env.set_state(np.array([i], dtype=int))
                            ns, r, done, _ = self.env.step(np.array([j], dtype=int))
                        if not done:
                            candi.append(r + self.gamma * self.policy.v_table[ns])
                        else:
                            candi.append(r)
                    self.policy.policy_table[i] = self.policy.arg_max(np.array(candi))
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

    def reset(self, env):
        self._setup_model(env)

    def set_env(self, env):
        self.env = env

    def get_vec_normalize_env(self):
        return None

    def save(self, log_dir):
        with open(log_dir + ".tmp", "wb") as f:
            pickle.dump(self, f)
        os.replace(log_dir + ".tmp", log_dir + ".pkl")
