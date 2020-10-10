import sys
import math, gym
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
# import torchvision.transforms as T
from torch.autograd import Variable

pi = Variable(torch.FloatTensor([math.pi]))

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        action *= 0.5 * (self.action_space.high - self.action_space.low)
        return np.clip(action, self.action_space.low, self.action_space.high)

def normal(x, mu, sigma_sq):
    a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        self.linear2_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)

        return mu, sigma_sq


class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.model = self.model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def select_action(self, state):
        mu, sigma_sq = self.model(Variable(state))
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        # calculate the probability
        action = (mu + sigma_sq.sqrt() * Variable(eps)).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5 * ((sigma_sq + 2 * pi.expand_as(sigma_sq)).log() + 1)

        log_prob = prob.log()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i] * (Variable(R).expand_as(log_probs[i]))).sum() - (
                        0.0001 * entropies[i]).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()

if __name__ == "__main__":
    print("Start REINFORCE Algorithm by chingyaoc")
    env = NormalizedActions(gym.make("CartPoleCont-v0"))
    agent = REINFORCE(16, 4, env.action_space)
    for i in range(6000):
        state = torch.Tensor([env.reset()])
        entropies = []
        log_probs = []
        rewards = []
        for t in range(1000):
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.numpy()[0])
            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = torch.Tensor([next_state])
            if done:
                break
        agent.update_parameters(rewards, log_probs, entropies, 0.99)
        if (i+1)%100 == 0:
            print("Now, %dth iterations" %(i+1))
            print("current episode length is %d" % (t))

    torch.save(agent.model.state_dict(), "./model_parameters_other.pt")

    state = torch.Tensor([env.reset()])
    for i in range(1000):
        env.render("human")
        action, _, _ = agent.select_action(state)
        ns, _, done, _ = env.step(action.numpy()[0])
        state = torch.Tensor([ns])
        if done:
            break
    env.close()