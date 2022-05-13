import gym
import random
import numpy as np
import torch as th
from typing import Optional, Tuple


class TabularPolicy:
    def __init__(
            self,
            observation_space,
            action_space,
            env,
            epsilon: float = 0.3,
            alpha: float = 1.0,
            beta: float = 0.5,
            device: str = 'cpu',
            **kwargs,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.init_kwargs = kwargs
        self._setup_table(**kwargs)

    def _setup_table(self, **kwargs):
        if isinstance(self.observation_space, gym.spaces.MultiDiscrete):
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
        elif hasattr(self.env, "get_vectorized") and callable(getattr(self.env, "get_vectorized")):
            s_vec, a_vec = self.env.get_vectorized()
            self.dim = self.observation_space.shape[0]
            self.obs_size = len(s_vec)
            self.act_size = len(a_vec)
        self.q_table = th.zeros([self.act_size, self.obs_size], dtype=th.float32).to(self.device)
        self.v_table = th.full([self.obs_size], 0, dtype=th.float32).to(self.device)
        self.policy_table = th.full([self.act_size, self.obs_size], 1 / self.act_size, dtype=th.float32).to(self.device)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
        obs_idx = self.env.get_idx_from_obs(observation)
        if not deterministic:
            act_idx = self.choice_act(self.policy_table.T[obs_idx])
        else:
            act_idx = self.arg_max(self.policy_table.T[obs_idx])
        action = self.env.get_acts_from_idx(act_idx)
        return action, None

    def forward(self, observation, deterministic=False):
        return self.predict(observation, deterministic=deterministic)

    def reset(self, **kwargs):
        if not kwargs:
            kwargs = self.init_kwargs
        self._setup_table(**kwargs)

    def get_log_prob_from_act(self, obs, acts):
        obs_idx = self.env.obs_to_idx(obs)
        acts_idx = self.env.act_to_idx(acts)
        probs = self.policy_table[obs_idx]
        log_probs = np.log(probs[range(len(acts_idx)), acts_idx])
        return th.from_numpy(log_probs).float()

    def arg_max(self, x):
        arg = []
        for x_ in x:
            arg.append(random.choice(np.flatnonzero((x_ == x_.max()).cpu().numpy())))
        return np.array(arg)

    def choice_act(self, policy):
        arg = []
        for prob in policy:
            arg.append(random.choices(range(self.act_size), weights=prob)[0])
        return np.array(arg, dtype=int)


class FiniteTabularPolicy(TabularPolicy):
    def _setup_table(self, **kwargs):
        max_t = kwargs.pop('max_t')
        super(FiniteTabularPolicy, self)._setup_table(**kwargs)
        self.policy_table = self.policy_table.repeat(max_t, 1, 1)
        self.q_table = self.q_table.repeat(max_t, 1, 1)
        self.v_table = self.v_table.repeat(max_t, 1)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError
