import random
import numpy as np
import torch as th
from typing import Optional, Tuple


class TabularPolicy:
    def __init__(
            self,
            observation_space,
            action_space,
            epsilon: float = 0.3,
            alpha: float = 1.0,
            beta: float = 0.5,
            device: str = 'cpu',
            **kwargs,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.init_kwargs = kwargs
        self._setup_table(**kwargs)

    def _setup_table(self, **kwargs):
        self.map_size = self.observation_space.nvec[0]
        self.act_size = self.action_space.nvec[0]
        if len(self.observation_space.nvec) == 1:
            self.dim = 1
            self.obs_size = self.map_size
        elif len(self.observation_space.nvec) == 2:
            self.dim = 2
            self.obs_size = self.map_size * self.observation_space.nvec[1]
            self.act_size *= self.action_space.nvec[1]
        else:
            raise NotImplementedError
        self.q_table = np.zeros([self.obs_size, self.act_size])
        self.v_table = np.full([self.obs_size], -np.inf)
        self.policy_table = np.full([self.obs_size, self.act_size], 1 / self.act_size)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
        obs_idx = self.obs_to_idx(observation)
        if not deterministic:
            act_idx = self.choice_act(self.policy_table[obs_idx])
        else:
            act_idx = self.arg_max(self.policy_table[obs_idx])
        action = self.idx_to_act(act_idx)
        return action, None

    def forward(self, observation, deterministic=False):
        return self.predict(observation, deterministic=deterministic)

    def reset(self, **kwargs):
        if not kwargs:
            kwargs = self.init_kwargs
        self._setup_table(**kwargs)

    def get_log_prob_from_act(self, obs, acts):
        obs_idx = self.obs_to_idx(obs)
        acts_idx = self.act_to_idx(acts)
        probs = self.policy_table[obs_idx]
        log_probs = np.log(probs[range(len(acts_idx)), acts_idx])
        return th.from_numpy(log_probs).float()

    def arg_max(self, x):
        enum, anum = x.shape
        arg = np.zeros([enum, 1], dtype=int)
        for i, x_ in enumerate(x):
            arg[i] = random.choice(np.flatnonzero(x_ == x_.max()))
        return arg

    def choice_act(self, policy):
        arg = []
        for prob in policy:
            arg.append([random.choices(range(self.act_size), weights=prob)[0]])
        return np.array(arg, dtype=int)

    def obs_to_idx(self, obs: np.ndarray) -> np.ndarray:
        if self.dim == 1:
            obs_idx = obs.reshape(-1)
        elif self.dim == 2:
            obs_idx = obs[:, 0] + self.map_size * obs[:, 1]
        else:
            raise NotImplementedError
        return obs_idx

    def idx_to_obs(self, idx: np.ndarray) -> np.ndarray:
        idx = idx.reshape(-1, 1)
        if self.dim == 1:
            obs = idx
        elif self.dim == 2:
            obs = np.append(idx % self.map_size, idx // self.map_size, axis=-1)
        else:
            raise NotImplementedError
        return obs

    def act_to_idx(self, act: np.ndarray) -> np.ndarray:
        if self.dim == 1:
            act_idx = act.reshape(-1)
        elif self.dim == 2:
            act_idx = act[:, 0] + self.action_space.nvec[0] * act[:, 1]
        else:
            raise NotImplementedError
        return act_idx

    def idx_to_act(self, idx: np.ndarray) -> np.ndarray:
        idx = idx.reshape(-1, 1)
        if self.dim == 1:
            act = idx
        elif self.dim == 2:
            act = np.append(idx % self.action_space.nvec[0], idx // self.action_space.nvec[0], axis=-1)
        else:
            raise NotImplementedError
        return act


class FiniteTabularPolicy(TabularPolicy):
    def _setup_table(self, **kwargs):
        max_t = kwargs.pop('max_t')
        super(FiniteTabularPolicy, self)._setup_table(**kwargs)
        self.policy_table = np.repeat(self.policy_table[None, :], max_t, axis=0)
        self.q_table = np.repeat(self.q_table[None, :], max_t, axis=0)
        self.v_table = np.repeat(self.v_table[None, :], max_t, axis=0)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError
