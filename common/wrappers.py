import gym
import numpy as np
import torch


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)
        self.coeff = 50
        self.action_space = gym.spaces.Box(
            low=self.env.action_space.low / self.coeff,
            high=self.env.action_space.high / self.coeff,
            dtype=np.float64
        )

    def action(self, action):
        return self.coeff * action

    def reverse_action(self, action):
        return 1/(action * self.coeff)


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self._ob = None
        self.rwfn = rwfn
        self.use_action_as_inp = self.rwfn.use_action_as_inp

    def train(self, mode=True):
        self.rwfn.train(mode)

    def eval(self):
        self.train(mode=False)

    def step(self, action: np.ndarray):
        assert self._ob is not None
        inp = self._ob.reshape(1, -1)
        if self.use_action_as_inp:
            inp = np.append(inp, action.reshape(1, -1), axis=1)
        r = self.reward(inp).item()
        self._ob, _, done, info = self.env.step(action)
        return self._ob, r, done, info

    def reset(self, **kwargs):
        self._ob = self.env.reset(**kwargs)
        return self._ob

    def get_reward_mat(self):
        inp, acts = self.env.get_vectorized()
        if self.use_action_as_inp:
            r = torch.zeros([len(acts), len(inp)]).to(self.rwfn.device)
            for i, act in enumerate(acts):
                r[i, :] = self.reward(np.append(inp, np.repeat(act[None, :], len(inp), axis=0), axis=1))
        else:
            r = self.reward(inp)
        if hasattr(self.env, 'get_done_mat'):
            r -= torch.from_numpy(self.env.get_done_mat() * 50).float().to(self.rwfn.device)
        return r

    def reward(self, inp: np.ndarray) -> torch.Tensor:
        rwinp = torch.from_numpy(inp).float().to(self.rwfn.device)
        return self.rwfn.forward(rwinp).squeeze()


class RewardInputNormalizeWrapper(RewardWrapper):
    def reward(self, inp: np.ndarray) -> torch.Tensor:
        if isinstance(self.observation_space, gym.spaces.MultiDiscrete):
            high = (self.observation_space.nvec[None, :] - 1)
        elif isinstance(self.observation_space, gym.spaces.Box):
            candi = np.append(abs(self.observation_space.high[None, :]), abs(self.observation_space.low[None, :]), axis=0)
            high = np.max(candi, axis=0, keepdims=True)
        else:
            raise NotImplementedError
        if self.use_action_as_inp:
            if isinstance(self.action_space, gym.spaces.MultiDiscrete):
                high = np.append(high, (self.action_space.nvec[None, :] - 1), axis=-1)
            elif isinstance(self.action_space, gym.spaces.Box):
                candi = np.append(abs(self.action_space.high[None, :]), abs(self.action_space.low[None, :]), axis=0)
                high = np.append(high, np.max(candi, axis=0, keepdims=True), axis=-1)
            else:
                raise NotImplementedError
        inp = inp / high
        return super().reward(inp)


class ActionNormalizeRewardWrapper(RewardWrapper):
    def __init__(self, env, rwfn):
        super(ActionNormalizeRewardWrapper, self).__init__(env, rwfn)
        self.coeff = 100

    def reward(self, inp: np.ndarray) -> torch.tensor:
        if self.use_action_as_inp:
            n_acts = self.action_space.shape[0]
            inp[:, -n_acts:] = inp[:, -n_acts:] / self.coeff
        return super().reward(inp)


class DiscretizeWrapper(gym.Wrapper):
    def step(self, action):
        n_obs, reward, done, info = self.env.step(action)
        d_n_obs = self.env.get_obs_from_idx(self.env.get_idx_from_obs(n_obs)).squeeze()
        self.env.set_state(d_n_obs)
        return d_n_obs, reward, done, info

    def reset(self):
        state = self.env.reset()
        d_state = self.env.get_obs_from_idx(self.env.get_idx_from_obs(state)).squeeze()
        self.env.set_state(d_state)
        return d_state
