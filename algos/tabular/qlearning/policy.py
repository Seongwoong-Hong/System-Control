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

    def arg_max(self, x):
        maxi = -np.inf
        idx = []
        for i, data in enumerate(x.flatten()):
            if data > maxi:
                idx = [i]
                maxi = data
            elif data == maxi:
                idx.append(i)
                maxi = data
        arg_max = random.sample(idx, 1)[0]
        return arg_max


class TabularSoftPolicy(TabularPolicy):
    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        action = np.zeros(observation.shape, dtype=int)
        for i, ob in enumerate(observation):
            if not deterministic:
                action[i] = self.arg_softmax(self.q_table[ob, :])
            else:
                action[i] = self.arg_max(self.q_table[ob, :])
        return action, None

    def softmax(self, x):
        y = np.exp(x - np.max(x))
        f_x = y / np.sum(y)
        return f_x

    def arg_softmax(self, x):
        arg_probs = self.softmax(x)
        arg = random.choices(range(len(x)), weights=arg_probs)[0]
        return arg
