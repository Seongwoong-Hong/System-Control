import torch, gym, math, copy
import numpy as np
from torch import nn, optim
from torch.distributions import Normal

class NormalizedWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        return observation, self.reward(reward), done, info

    def action(self, action):
        action *= 0.5 * (self.action_space.high - self.action_space.low)
        return np.clip(action, self.action_space.low, self.action_space.high)

    def reward(self, reward):
        return reward ** 0.25

class policy_model(nn.Module):
    def __init__(self, env):
        super(policy_model, self).__init__()
        self.ob_space = env.observation_space.shape[0]
        self.a_space = env.action_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(self.ob_space, 64),
            nn.ReLU(),
            nn.Linear(64, self.a_space),
        )
        self.log_std = nn.Parameter(torch.ones(self.a_space))

    def forward(self, x, a=None, use_a=False):
        mu = self.network(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        if not use_a:
            a = dist.sample()
        log_prob = dist.log_prob(a)
        entropy = 0.5 * torch.log(2 * math.pi * std ** 2) + 1
        return a, log_prob, entropy
# class REINFORCE:
#     def __init__(self, env, policy):

def rollout(n, pi, env, gamma=1):
    ep_r = []
    ep_x, ep_a = np.zeros([4, 1, n]), np.zeros([n])
    x = env.reset().reshape(4, 1)
    for i in range(n):
        ep_x[:, :, i] = x
        a, _, _ = pi.forward(torch.Tensor(x).squeeze())
        ob, r, done, _ = env.step(np.array([a.item()]))
        x = ob.reshape(4, 1)
        ep_r.append(r)
        ep_a[i] = a
        if done:
            n = i + 1
            ep_r = ep_r[0:n]
            ep_x = ep_x[:, :, 0:n]
            ep_a = ep_a[0:n]
            break
    ep_v = torch.zeros(n)
    prev_v = 0
    for i in range(n):
        idx = n - 1 - i
        ep_v[idx] = gamma * prev_v + ep_r[idx]
        prev_v = copy.deepcopy(ep_v[idx])
    # ep_v -= ep_v.mean()
    return ep_x, ep_a, ep_v

def lossfn(x, a, v):
    n = len(a)
    loss = 0
    for i in range(n):
        _, log_prob, entropy = pi.forward(torch.Tensor(x[:, :, i]).squeeze(), a[i], use_a=True)
        loss -= log_prob * v[i] - 0.001 * entropy
    return loss/n

def fill_batch(ep_x, ep_a, ep_v, batch_size, ep_cum_len):
    ep_cum_len_log, ep_count = 0, 0
    while ep_cum_len < batch_size:
        ep_x, ep_a, ep_v = rollout(batch_size, pi, env, gamma=0.96)
        ep_cum_len += len(ep_a)
        ep_cum_len_log += len(ep_a)
        ep_count += 1
        if ep_cum_len > batch_size:
            remain = batch_size - (ep_cum_len - len(ep_a))
            batch_x = np.concatenate([remain_x, ep_x[:, :, 0:remain]], 2)
            batch_a = np.concatenate([remain_a, ep_a[0:remain]])
            batch_v = np.concatenate([remain_v, ep_v[0:remain]])
            remain_x = ep_x[:, :, remain - 1:-1]
            remain_a = ep_a[remain - 1:-1]
            remain_v = ep_v[remain - 1:-1]
        else:
            remain_x = np.concatenate([remain_x, ep_x], 2)
            remain_a = np.concatenate([remain_a, ep_a])
            remain_v = np.concatenate([remain_v, ep_v])
    ep_cum_len -= batch_size

if __name__ == '__main__':
    print("REINFORCE Algorithm for continuous cartpole problem")
    env = NormalizedWrapper(gym.make("CartPoleCont-v0"))
    batch_size, epoch = 5000, 5000
    pi = policy_model(env)
    optimizer = optim.Adam(pi.parameters(), lr=0.001)
    pi.train()
    ep_tot_len, ep_tot_count, ep_cum_len = 0, 0, 0
    remain_x, remain_a, remain_v = np.empty((4, 1, 0)), np.empty(0), np.empty(0)
    for i in range(epoch):
        ep_cum_len_log, ep_count = 0, 0
        while ep_cum_len < batch_size:
            ep_x, ep_a, ep_v = rollout(batch_size, pi, env, gamma=0.99)
            ep_cum_len += len(ep_a)
            ep_cum_len_log += len(ep_a)
            ep_count += 1
            if ep_cum_len > batch_size:
                remain = batch_size - (ep_cum_len - len(ep_a))
                batch_x = np.concatenate([remain_x, ep_x[:, :, 0:remain]], 2)
                batch_a = np.concatenate([remain_a, ep_a[0:remain]])
                batch_v = np.concatenate([remain_v, ep_v[0:remain]])
                remain_x = ep_x[:, :, remain - 1:-1]
                remain_a = ep_a[remain - 1:-1]
                remain_v = ep_v[remain - 1:-1]
            else:
                remain_x = np.concatenate([remain_x, ep_x], 2)
                remain_a = np.concatenate([remain_a, ep_a])
                remain_v = np.concatenate([remain_v, ep_v])
        ep_cum_len -= batch_size
        optimizer.zero_grad()
        # loss = lossfn(batch_x, batch_a, batch_v)
        loss = lossfn(batch_x, batch_a, batch_v)
        loss.backward()
        optimizer.step()
        ep_tot_len += ep_cum_len_log
        ep_tot_count += ep_count
        if (i + 1) % 10 == 0:
            print("--------------------------------------")
            print("Iter                     |%d" % (i + 1))
            print("Current Loss             |%.5f" % (loss.item()))
            print("Initial State Value      |%.2f" % (ep_v[0]))
            print("Final State Value        |%.2f" % (ep_v[-1]))
            print("Average episode length   |%.2f" % (ep_tot_len/ep_tot_count))
            print("--------------------------------------")
            ep_tot_len, ep_tot_count = 0, 0

    torch.save(pi.state_dict(), "./model_parameters.pt")