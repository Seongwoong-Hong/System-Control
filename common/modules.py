from typing import List, Type

import sys
import random
import torch
from torch import nn

from stable_baselines3.common.logger import Logger, HumanOutputFormat


class CostNet(nn.Module):
    def __init__(self,
                 arch: List[int],
                 device: str,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 act_fcn: Type[nn.Module] = None,
                 lr: float = 3e-5,
                 num_expert: int = 10,
                 num_samp: int = 10,
                 num_act: int = 1,
                 decay_coeff: float = 0.0,
                 verbose: int = 1,
                 ):
        """
        :param arch: The architecture of a Neural Network form of the cost function.
            The first argument of the arch must be an input size.
        :param device: The device(cpu or cuda) you want to use.
        :param optimizer_class: The optimizer for stochastic gradient descent.
        :param act_fcn: The activation function for each layers of the cost function.
        :param lr: learning rate
        :param num_expert: How many expert trajectories you use when optimizing cost function.
        :param num_samp: How many sample trajectories from current policy you us when optimizing cost function.
        :param num_act: Number of actions of your current environment.
        :param decay_coeff: The coefficient for preventing parameter decaying loss
        :param verbose: The verbosity level. 0: no outputs, 1: info
        """
        super(CostNet, self).__init__()
        self.device = device
        self.optimizer_class = optimizer_class
        self.act_fnc = act_fcn
        self.evalmod = False
        self.num_expert = num_expert
        self.num_samp = num_samp
        self.decay_coeff = decay_coeff
        self.num_act = num_act
        self.sampleE = [None]
        self.sampleL = [None]
        self.verbose = verbose
        self._build(lr, arch)

    def _build(self, lr, arch):
        # configure neural net layers for the cost function
        layers = []
        if self.act_fnc is not None:
            for i in range(len(arch)-1):
                layers.append(nn.Linear(arch[i], arch[i+1]))
                layers.append(self.act_fnc())
        else:
            for i in range(len(arch)-1):
                layers.append(nn.Linear(arch[i], arch[i+1]))
        layers.append(nn.Linear(arch[-1], 1, bias=False))
        self.layers = nn.Sequential(*layers)
        self.aparam = nn.Linear(self.num_act, 1, bias=False)
        self.optimizer = self.optimizer_class(self.parameters(), lr)

    def forward(self, obs):
        if self.evalmod:
            with torch.no_grad():
                return self.layers(obs[:, :-self.num_act])**2 + self.aparam(obs[:, -self.num_act:])**2
        else:
            return self.layers(obs[:, :-self.num_act])**2 + self.aparam(obs[:, -self.num_act:])**2

    def _setup_learn(self, learner_trans, expert_trans):
        self.train_()
        logger = None
        # configure logger for logging
        if self.verbose >= 1:
            logger = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout)])
        # sample trajectories that will use for learning
        sampleE = random.sample(expert_trans, self.num_expert)
        sampleL = random.sample(learner_trans, self.num_samp)
        return logger, sampleE, sampleL

    def learn(self, learner_trans, expert_trans, epoch: int = 10):
        logger, sampleE, sampleL = self._setup_learn(learner_trans, expert_trans)
        IOCLoss1, IOCLoss2, lcrLoss, monoLoss, paramLoss = None, None, None, None, None
        for _ in range(epoch):
            IOCLoss1, IOCLoss2, lcrLoss, monoLoss, paramLoss = 0, 0, 0, 0, 0
            # prevent parameter decaying
            param_norm = 0
            for param in self.parameters():
                param_norm += torch.norm(param)
            paramLoss = self.decay_coeff * torch.max(torch.zeros(1).to(self.device), (1 - torch.norm(param_norm)))

            # Calculate learned cost loss
            for E_trans in sampleE:
                costs = self.forward(E_trans[:, :-1])
                monoLoss += torch.sum(torch.max(torch.zeros(*costs.shape[:-1], device=self.device),
                                                costs[1:] - costs[:-1] - 1) ** 2)
                IOCLoss1 += costs
            IOCLoss1 = torch.mean(IOCLoss1)

            # Calculate Max Ent. Loss
            x = torch.zeros(len(sampleE+sampleL)).double().to(self.device)
            for j, trans_j in enumerate(sampleE + sampleL):
                costs = self.forward(trans_j[:, :-1])
                x[j] = -torch.sum(costs + trans_j[:, -1])
                lcrLoss += torch.sum((costs[2:] - 2*costs[1:-1] + costs[:-2]) ** 2)
            IOCLoss2 = -torch.logsumexp(x, 0)

            IOCLoss = IOCLoss1 + IOCLoss2 + monoLoss + lcrLoss + paramLoss
            self.optimizer.zero_grad()
            IOCLoss.backward()
            self.optimizer.step()
        if self.verbose >= 1:
            exclude = ("tensorboard", "json", "csv")
            logger.record("Cost/Expert_Cost_loss", IOCLoss1.item(), exclude=exclude)
            logger.record("Cost/Max_Ent._Cost_loss", IOCLoss2.item(), exclude=exclude)
            logger.record("Cost/Mono_Regularization", monoLoss.item(), exclude=exclude)
            logger.record("Cost/lcr_Regularization", lcrLoss.item(), exclude=exclude)
            logger.record("Cost/param_Regularization", paramLoss.item(), exclude=exclude)
            logger.dump()
        return self

    def train_(self):
        self.evalmod = False
        return self.train()

    def eval_(self):
        self.evalmod = True
        return self.eval()
