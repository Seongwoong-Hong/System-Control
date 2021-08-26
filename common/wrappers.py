import gym
import numpy as np
import torch


class ActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return 0.4 * action

    def reverse_action(self, action):
        return 1/action


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn
        self.use_action_as_inp = self.rwfn.use_action_as_inp

    def step(self, action: np.ndarray):
        ob, rew, done, info = self.env.step(action)
        inp = info['obs']
        if self.use_action_as_inp:
            inp = np.append(inp, info['acts'], axis=1)
        return ob, self.reward(inp), done, info

    def reward(self, inp) -> float:
        rwinp = torch.from_numpy(inp).reshape(1, -1).to(self.rwfn.device)
        return self.rwfn.forward(rwinp).item()


class ActionRewardWrapper(RewardWrapper):
    def __init__(self, env, rwfn):
        super(ActionRewardWrapper, self).__init__(env, rwfn)
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


class ObsConcatWrapper(gym.Wrapper):
    def __init__(self, env, num_timesteps):
        super(ObsConcatWrapper, self).__init__(env)
        self.num_timesteps = num_timesteps
        self.obs_list = np.zeros([self.num_timesteps, self.observation_space.shape[0]])
        self.acts_list = np.zeros([self.num_timesteps, self.action_space.shape[0]])

    def step(self, action: np.ndarray):
        ob, rew, done, info = self.env.step(action)
        self.acts_list = np.append([action], self.acts_list[:-1], axis=0)
        info['obs'] = self.obs_list
        info['acts'] = self.acts_list
        self.obs_list = np.append([ob], self.obs_list[:-1], axis=0)
        return ob, rew, done, info

    def reset(self):
        ob = super(ObsConcatWrapper, self).reset()
        self.obs_list = np.zeros([self.num_timesteps, self.observation_space.shape[0]])
        self.acts_list = np.zeros([self.num_timesteps, self.action_space.shape[0]])
        self.obs_list[0, :] = ob
        return ob
