import gym
import numpy as np
import torch


class ActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return 0.4 * action

    def reverse_action(self, action):
        return 1/action


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn, timesteps: int = 1):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn
        self.use_action_as_inp = self.rwfn.use_action_as_inp
        self.timesteps = timesteps
        if self.use_action_as_inp:
            self.inp_list = np.zeros([self.timesteps, env.current_obs.shape[0] + env.action_space.shape[0]])
        else:
            self.inp_list = np.zeros([self.timesteps, env.current_obs.shape[0]])

    def step(self, action: np.ndarray):
        observation, _, done, info = self.env.step(action)
        if self.use_action_as_inp:
            inp = self.concat(np.append(info['rw_inp'], action))
        else:
            inp = self.concat(info['rw_inp'])
        return observation, self.reward(inp), done, info

    def reward(self, inp: np.ndarray) -> float:
        rwinp = torch.from_numpy(inp).reshape(1, -1).to(self.rwfn.device)
        return self.rwfn.forward(rwinp).item()

    def concat(self, inp: np.ndarray) -> np.ndarray:
        self.inp_list = np.append(self.inp_list[-self.timesteps + 1:], [inp], axis=0)
        return self.inp_list.reshape(-1)


class ActionRewardWrapper(RewardWrapper):
    def __init__(self, env, rwfn, timesteps: int = 1):
        super(ActionRewardWrapper, self).__init__(env, rwfn, timesteps)
        self.clip_coeff = 0.4

    def action(self, action):
        return self.clip_coeff * action

    def step(self, action):
        return super(ActionRewardWrapper, self).step(self.action(action))


class CostWrapper(gym.RewardWrapper):
    def __init__(self, env, costfn):
        super(CostWrapper, self).__init__(env)
        self.costfn = costfn

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(np.append(observation, action)), done, info

    def reward(self, obs):
        cost_inp = torch.from_numpy(obs).reshape(1, -1).to(self.costfn.device)
        return -self.costfn.forward(cost_inp)


class ActionCostWrapper(gym.RewardWrapper):
    def __init__(self, env, costfn):
        super(ActionCostWrapper, self).__init__(env)
        self.gear = env.model.actuator_gear[0, 0]
        self.costfn = costfn

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(self.action(action))
        return observation, self.reward(np.append(observation, action)), done, info

    def action(self, action: np.ndarray):
        return self.gear * action

    def reverse_action(self, action: np.ndarray):
        return action / self.gear

    def reward(self, obs):
        cost_inp = torch.from_numpy(obs).reshape(1, -1).to(self.costfn.device)
        return -self.costfn.forward(cost_inp)
