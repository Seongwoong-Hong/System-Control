import gym
import numpy as np
import torch

from typing import Callable


class ActionRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(ActionRewardWrapper, self).__init__(env)
        self.rwfn = rwfn

    # noinspection PyMethodMayBeStatic
    def action(self, action):
        new_action = 2 * action
        return new_action

    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        return observation, self.reward(np.append(observation, self.action(action))), done, info

    def reward(self, obs):
        rwinp = torch.from_numpy(obs).reshape(1, -1).to(self.rwfn.device)
        return self.rwfn.forward(rwinp)


class ActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return action

    def reverse_action(self, action):
        return 1/action


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn
        self.use_action_as_inp = self.rwfn.use_action_as_inp

    def step(self, action: np.ndarray):
        obs = self.env.current_obs
        observation, _, done, info = self.env.step(action)
        if self.use_action_as_inp:
            return observation, self.reward(np.append(obs, action)), done, info
        else:
            return observation, self.reward(obs), done, info

    def reward(self, inp):
        rwinp = torch.from_numpy(inp).reshape(1, -1).to(self.rwfn.device)
        return self.rwfn.forward(rwinp).item()


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
