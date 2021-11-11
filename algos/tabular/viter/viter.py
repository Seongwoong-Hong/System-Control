import gym
import numpy as np
from copy import deepcopy
from algos.tabular.viter.policy import TabularPolicy

class Viter:
    def __init__(
            self,
            env,
            gamma,
            epsilon
    ):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        assert isinstance(self.observation_space, gym.spaces.MultiDiscrete) \
               and isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.gamma = gamma
        self.policy = TabularPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            epsilon=epsilon,
        )

    def train(self, reward_table):
        for i in range(self.observation_space.nvec[0]):
            candi = []
            for j in range(self.action_space.nvec[0]):
                candi.append(self.policy.v_table[i] + self.gamma * reward_table[i][j])
            self.policy.v_table[i] = np.max(candi)

    def learn(self, train_iter):
        self.env.reset()
        reward_table = np.zeros([self.observation_space.nvec[0], self.action_space.nvec[0]])
        for i in range(self.observation_space.nvec[0]):
            for j in range(self.action_space.nvec[0]):
                self.env.set_state(np.array([i], dtype=int))
                ns, r, done, _ = self.env.step(np.array([j], dtype=int))
                reward_table[i][j] = r
        for i in range(train_iter):
            old_value = deepcopy(self.policy.v_table)
            self.train(reward_table)
            error = np.mean(np.abs(old_value - self.policy.v_table))
            print(f"{i}th iter: Mean difference btw old value and current value is {error:.3f}")
            if error < 0.001:
                break
        for i in range(self.observation_space.nvec[0]):
            candi = []
            for j in range(self.action_space.nvec[0]):
                self.env.set_state(np.array([i], dtype=int))
                ns, _, done, _ = self.env.step(np.array([j], dtype=int))
                candi.append(reward_table[i][j] + self.gamma * self.policy.v_table[ns])
            self.policy.policy_table[i] = np.argmax(candi)
