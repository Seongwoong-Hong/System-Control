import gym
import pickle
import numpy as np

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
    ):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        assert isinstance(self.observation_space, gym.spaces.MultiDiscrete) and isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.device = device
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
            self.policy.q_table[ob.item(), action.item()] += \
                self.alpha * (reward + self.gamma * np.max(self.policy.q_table[next_ob.item(), :])
                              - self.policy.q_table[ob.item(), action.item()])
        else:
            self.policy.q_table[ob.item(), action.item()] += self.alpha * (reward - self.policy.q_table[ob.item(), action.item()])

    def learn(
            self,
            train_length,
    ) -> None:
        prev_q_table = deepcopy(self.policy.q_table + 1)
        for t in range(train_length):
            ob = self.env.reset()
            done = False
            while not done:
                act, _ = self.policy.predict(ob, deterministic=False)
                next_ob, reward, done, _ = self.env.step(act)
                self.train(ob, act, next_ob, reward, done)
                ob = next_ob
            for i in range(self.observation_space.nvec[0]):
                self.policy.policy_table[i] = np.argmax(self.policy.q_table[i])
            error = np.mean(np.abs(self.policy.q_table - prev_q_table))
            print(f"Mean difference btw previous and current q is {error:.3f}")
            if error < 1e-3 and t > 100:
                break
            prev_q_table = deepcopy(self.policy.q_table)

    def get_env(self):
        return self.env

    def set_env(self, env):
        self.env = env

    def predict(self, observation, deterministic):
        return self.policy.predict(observation, deterministic)

    @classmethod
    def load(
        cls,
        path,
        env,
        device,
    ):
        pass

    def save(
        self,
        path,
    ):
        pass

