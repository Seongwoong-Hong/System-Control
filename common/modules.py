import torch, random, time
from torch import nn
from typing import List

class NNCost(nn.Module):
    def __init__(self,
                 arch: List[int],
                 device='cpu',
                 optimizer_class=torch.optim.Adam,
                 act_fcn=nn.Tanh,
                 lr: float = 3e-5,
                 num_expert: int = 10,
                 num_samp: int = 10,
                 num_act: int = 1,
                 decay_coeff: float = 0.0):
        """
        :param arch: The architecture of a Neural Network form of the cost function.
            The first argument of the arch must be an input size.
        :param device: The device(cpu or cuda) you want to use.
        :param optimizer: The optimizer for stochastic gradient descent.
        :param act_fcn: The activation function for each layers of the cost function.
        :param lr: learning rate
        :param num_expert: How many expert trajectories you use when optimizing cost function.
        :param num_samp: How many sample trajectories from current policy you us when optimizing cost function.
        :param num_act: Number of actions of your current environment.
        :param decay_coeff: The coefficient for preventing parameter decaying loss
        """
        super(NNCost, self).__init__()
        self.device = device
        self.optimizer_class = optimizer_class
        self.act_fnc = act_fcn
        self.evalmod = False
        self.num_expert = num_expert
        self.num_samp = num_samp
        self.decay_coeff = decay_coeff
        self.num_act = num_act

        self._build(lr, arch)

    def _build(self, lr, arch):
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            layers.append(self.act_fnc())
        layers.append(nn.Linear(arch[-1], 1, bias=True))
        self.layers = nn.Sequential(*layers)
        self.aparam = nn.Linear(self.num_act, 1, bias=False)
        self.optimizer = self.optimizer_class(self.parameters(), lr)

    def sample_trajectory_sets(self, learner_trans, expert_trans):
        self.sampleE = random.sample(expert_trans, self.num_expert)
        self.sampleL = random.sample(learner_trans, self.num_samp)

    def forward(self, obs):
        if self.evalmod:
            with torch.no_grad():
                return self.layers(obs[:-self.num_act])**2 + self.aparam(obs[-self.num_act:])**2
        else:
            return self.layers(obs[:-self.num_act])**2 + self.aparam(obs[-self.num_act:])**2

    def learn(self, epoch: int):
        self._train()
        for _ in range(epoch):
            IOCLoss1, IOCLoss2, paramLoss = 0, 0, 0
            # Calculate the loss for preventing parameter decaying
            param_norm = 0
            for param in self.parameters():
                param_norm += torch.norm(param)
            paramLoss = self.decay_coeff * (1 - torch.norm(param_norm))

            # Calculate learned cost loss
            for E_trans in self.sampleE:
                for info in E_trans.infos:
                    IOCLoss1 += self.forward(info['rwinp'])
            IOCLoss1 /= len(self.sampleE)

            # Calculate Max Ent. Loss
            x = torch.zeros(len(self.sampleE+self.sampleL)).double()
            for j, trans_j in enumerate(self.sampleE+self.sampleL):
                temp = 0
                for info in trans_j.infos:
                    temp -= self.forward(info['rwinp'])+info['log_probs']
                x[j] = temp
            IOCLoss2 = -torch.logsumexp(x, 0)

            IOCLoss = IOCLoss1 + IOCLoss2 + paramLoss
            self.optimizer.zero_grad()
            IOCLoss.backward()
            self.optimizer.step()
        print("Loss for Expert cost: {:.2f}, Loss for Max Ent.: {:.2f}".format(IOCLoss1.item(), IOCLoss2.item()))
        return self

    def _train(self):
        self.evalmod = False
        return self.train()

    def _eval(self):
        self.evalmod = True
        return self.eval()