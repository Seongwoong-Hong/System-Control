import gym
import numpy as np
import torch


class ActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return 0.4 * action

    def reverse_action(self, action):
        return 1/(action * 0.4)


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn
        self.use_action_as_inp = self.rwfn.use_action_as_inp

    def train(self, mode=True):
        self.rwfn.train(mode)

    def eval(self):
        self.train(mode=False)

    def step(self, action: np.ndarray):
        next_ob, rew, done, info = self.env.step(action)
        inp = info['obs']
        if self.use_action_as_inp:
            inp = np.append(inp, info['acts'], axis=1)
        r = self.reward(inp).item()
        if info.get('done'):
            r -= 1000
        return next_ob, r, done, info

    def get_reward_mat(self):
        inp, a_vec = self.env.get_vectorized()
        if self.use_action_as_inp:
            acts = self.env.get_torque(a_vec).squeeze().T
            if hasattr(self, "action") and callable(self.action):
                acts = self.action(acts)
            r = torch.zeros([len(acts), len(inp)]).to(self.rwfn.device)
            for i, act in enumerate(acts):
                r[i] = self.reward(np.append(inp, np.repeat(act[None, :], len(inp), axis=0), axis=1))
        else:
            r = self.reward(inp)
        if self.rwfn.trainmode:
            return r
        else:
            return r.cpu().numpy()

    def reward(self, inp: np.ndarray) -> torch.tensor:
        rwinp = torch.from_numpy(inp).float().to(self.rwfn.device)
        return self.rwfn.forward(rwinp).squeeze()


class ActionRewardWrapper(RewardWrapper):
    def __init__(self, env, rwfn):
        super(ActionRewardWrapper, self).__init__(env, rwfn)
        self.coeff = 0.002

    def action(self, action):
        return self.coeff * action

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
