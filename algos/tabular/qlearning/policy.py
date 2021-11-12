import random
import numpy as np
from typing import Optional, Tuple

class TabularPolicy:
    def __init__(
            self,
            observation_space,
            action_space,
            epsilon,
            alpha
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_table = np.zeros([self.observation_space.nvec[0], self.action_space.nvec[0]])
        self.policy_table = np.zeros([self.observation_space.nvec[0]], dtype=int)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        action = self.policy_table[observation]
        if not deterministic:
            rd = random.random()
            if rd < self.epsilon:
                action = (self.action_space.np_random.random_sample(observation.shape) * self.action_space.nvec).astype(self.action_space.dtype)
        return action, None

    def forward(self, observation, deterministic=False):
        return self.predict(observation, deterministic=deterministic)
