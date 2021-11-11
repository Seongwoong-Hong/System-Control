import random
import numpy as np


class TabularPolicy:
    def __init__(
            self,
            observation_space,
            action_space,
            epsilon
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.v_table = np.zeros([self.observation_space.nvec[0]])
        self.policy_table = np.zeros([self.observation_space.nvec[0]], dtype=int)

    def predict(self, observation, deterministic=False):
        action = np.array([self.policy_table[observation.item()]], dtype=int)
        if not deterministic:
            rd = random.random()
            if rd < self.epsilon:
                action = self.action_space.sample()
        return action, None

    def forward(self, observation, deterministic=False):
        return self.predict(observation, deterministic)
