import torch, gym, math, copy
import numpy as np
from torch import nn, optim
from torch.autograd import Variable


class policy_model(nn.Module):
    def __init__(self, env):
        super(policy_model, self).__init__()
        self.ob_space = env.observation_space.shape[0]
        self.a_space = env.action_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(self.ob_space, 16),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(16, self.a_space)
        self.std_layer = nn.Linear(16, self.a_space)

    def sampling(self, mu, std):
        return mu + std * torch.randn_like(std)

    def log_normal(self, a, mu, std):
        return -0.5 * torch.log(2 * math.pi * std ** 2) - (a - mu) ** 2 / (2 * std ** 2)

    def forward(self, x):
        x = self.network(x)
        mu = self.mu_layer(x)
        std = self.std_layer(x)
        a = self.sampling(mu, std)
        log_prob = self.log_normal(Variable(a), mu, std)
        entropy = 0.5 * torch.log(2 * math.pi * std ** 2) + 1
        return a, log_prob, entropy

def rollout(n, pi, env, gamma=1):
    ep_r = []
    ep_log_prob = torch.zeros(n)
    ep_entropies = torch.zeros(n)
    x = env.reset().reshape(4, 1)
    for i in range(n):
        a, log_prob, entropy = pi.forward(torch.Tensor(x).squeeze())
        ob, r, done, _ = env.step(np.array([a.item()]))
        x = ob.reshape(4, 1)
        ep_r.append(r)
        ep_log_prob[i] = log_prob
        ep_entropies[i] = entropy
        if done:
            n = i + 1
            ep_r = ep_r[0:n]
            ep_log_prob = ep_log_prob[0:n]
            ep_entropies = ep_entropies[0:n]
            break

    ep_v = torch.zeros(n)
    prev_v = 0
    for i in range(n):
        idx = n - 1 - i
        ep_v[idx] = gamma * prev_v + ep_r[idx]
        prev_v = copy.deepcopy(ep_v[idx])

    return ep_v, ep_log_prob, ep_entropies

def loss_function(ep_v, ep_log_prob, ep_entropies):
    loss = -(ep_log_prob * ep_v + 0.01 * ep_entropies).mean()
    return loss

if __name__ == '__main__':
    print("REINFORCE Algorithm for continuous cartpole problem")
    env = gym.make("CartPoleCont-v0")
    n, ep_num = 1000, 100
    pi = policy_model(env)
    optimizer = optim.Adam(pi.parameters(), lr=0.001)
    pi.train()
    for i in range(ep_num):
        ep_v, ep_log_prob, ep_entropies = rollout(n, pi, env)
        optimizer.zero_grad()
        loss = loss_function(ep_v, ep_log_prob, ep_entropies)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print("Iter: %d" %(i + 1))
            print("Current Loss is: %.5f, Initial State Value is %.2f" %(loss.item(), ep_v[0]))

    ##### check #####
    x = env.reset().reshape(4, 1)
    pi.eval()
    env.render('human')
    for _ in range(n):
        a, _, _ = pi.forward(torch.Tensor(x).squeeze())
        ob, _, _, _ = env.step(np.array([a.item()]))
        x = ob.reshape(4, 1)
        env.render('human')
    env.close()

    torch.save(pi.state_dict(), "./model_parameters")