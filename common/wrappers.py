import gym, torch
import numpy as np

class ActionRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(ActionRewardWrapper, self).__init__(env)
        self.rwfn = rwfn

    def action(self, action):
        return action

    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        return observation, self.reward(np.append(observation, self.action(action))), done, info

    def reward(self, obs):
        rwinp = torch.from_numpy(obs).to(self.rwfn.device)
        return self.rwfn.forward(rwinp)

class ActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return action

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(np.append(observation, action)), done, info

    def reward(self, obs):
        rwinp = torch.from_numpy(obs).to(self.rwfn.device)
        return self.rwfn.forward(rwinp)


class CostWrapper(gym.RewardWrapper):
    def __init__(self, env, costfn):
        super(CostWrapper, self).__init__(env)
        self.costfn = costfn

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(np.append(observation, action)), done, info

    def reward(self, obs):
        cost_inp = torch.from_numpy(obs).to(self.costfn.device)
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
        cost_inp = torch.from_numpy(obs).to(self.costfn.device)
        return -self.costfn.forward(cost_inp)