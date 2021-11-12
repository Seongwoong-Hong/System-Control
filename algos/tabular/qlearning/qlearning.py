import os
import gym
import pickle
import numpy as np

from typing import Optional, Tuple
from copy import deepcopy

from algos.tabular.qlearning.policy import TabularPolicy


class QLearning:
    def __init__(
            self,
            env,
            gamma,
            epsilon,
            alpha,
            device,
            **kwargs,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.device = device
        self._setup_model(env)

    def _setup_model(self, env):
        self.num_timesteps = 0
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        assert isinstance(self.observation_space, gym.spaces.MultiDiscrete)\
               and isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.policy = TabularPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            epsilon=self.epsilon,
            alpha=self.alpha
        )

    def train(
            self,
            ob,
            action,
            next_ob,
            reward,
            done,
    ) -> None:
        if not done:
            self.policy.q_table[ob, action] += \
                self.alpha * (reward + self.gamma * np.max(self.policy.q_table[next_ob, :])
                              - self.policy.q_table[ob, action])
        else:
            self.policy.q_table[ob, action] += self.alpha * (reward - self.policy.q_table[ob, action])

    def learn(self, total_timesteps, **kwargs) -> None:
        t = 0
        while True:
            prev_q_table = deepcopy(self.policy.q_table)
            self.num_timesteps = t
            ob = self.env.reset()
            done = False
            while not done:
                act, _ = self.policy.predict(ob, deterministic=False)
                next_ob, reward, done, _ = self.env.step(act)
                self.train(ob, act, next_ob, reward, done)
                ob = next_ob
            for i in range(self.observation_space.nvec[0]):
                self.policy.policy_table[i] = np.argmax(self.policy.q_table[i])
            error = np.max(np.abs(self.policy.q_table - prev_q_table))
            if t % 10 == 0:
                print(f"{t}th iter: Max difference btw previous and current q is {error:.3f}")
            if error < 1e-8 and t > 100:
                print(f"{t}th iter: Max difference btw previous and current q is {error:.3f}")
                break
            t += 1

    def reset(self, env):
        self._setup_model(env)

    def get_env(self):
        return self.env

    def set_env(self, env):
        self.env = env

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
        with open(log_dir + ".tmp", "wb") as f:
            pickle.dump(self, f)
        os.replace(log_dir + ".tmp", log_dir + ".pkl")

