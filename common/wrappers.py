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
        return observation, self.reward(observation, self.action(action)), done, info

    def reward(self, observation, action):
        rwinp = torch.from_numpy(np.append(observation, action)).to(self.rwfn.device)
        return self.rwfn.forward(rwinp)

class ActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return action

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation, action), done, info

    def reward(self, observation, action):
        rwinp = torch.from_numpy(np.append(observation, action)).to(self.rwfn.device)
        return self.rwfn.forward(rwinp)


class CostWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(CostWrapper, self).__init__(env)
        self.rwfn = rwfn

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation, action), done, info

    def reward(self, observation, action):
        rwinp = torch.from_numpy(np.append(observation, action)).to(self.rwfn.device)
        return -self.rwfn.forward(rwinp)