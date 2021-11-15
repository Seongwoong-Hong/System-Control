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
        self.env.reset()
        for i in range(self.observation_space.nvec[0]):
            candi = []
            for j in range(self.action_space.nvec[0]):
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
                    self.env.reset()
            self.policy.v_table[i] = np.max(candi)

    def learn(self, total_timesteps, **kwargs):
        t = 0
        while t < total_timesteps:
            self.num_timesteps = t
            old_value = deepcopy(self.policy.v_table)
            self.train()
            error = np.max(np.abs(old_value - self.policy.v_table))
            logger.record("num_timesteps", t, exclude="tensorboard")
            logger.record("Value Error", error)
            logger.dump(t)
            print(f"{self.num_timesteps}th iter: Mean difference btw old value and current value is {error:.3f}")
            if error < 1e-8 and t > 10:
                for i in range(self.observation_space.nvec[0]):
                    candi = []
                    for j in range(self.action_space.nvec[0]):
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
                            self.env.reset()
                    self.policy.policy_table[i] = np.argmax(candi)
                break
            t += 1

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
