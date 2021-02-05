from typing import List

import random
import torch
from torch import nn

from stable_baselines3.common import logger, utils


class CostNet(nn.Module):
    def __init__(self,
                 arch: List[int],
                 device: str,
                 optimizer_class=torch.optim.Adam,
                 act_fcn=nn.Tanh,
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
        :param optimizer: The optimizer for stochastic gradient descent.
        :param act_fcn: The activation function for each layers of the cost function.
        :param lr: learning rate
        :param num_expert: How many expert trajectories you use when optimizing cost function.
        :param num_samp: How many sample trajectories from current policy you us when optimizing cost function.
        :param num_act: Number of actions of your current environment.
        :param decay_coeff: The coefficient for preventing parameter decaying loss
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
        # configure logger for logging
        utils.configure_logger(self.verbose)
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

    def sample_trajectory_sets(self, learner_trans, expert_trans):
        self.sampleE = random.sample(expert_trans, self.num_expert)
        self.sampleL = random.sample(learner_trans, self.num_samp)

    def forward(self, obs):
        if self.evalmod:
            with torch.no_grad():
                return self.layers(obs[:-self.num_act])**2 + self.aparam(obs[-self.num_act:])**2
        else:
            return self.layers(obs[:-self.num_act])**2 + self.aparam(obs[-self.num_act:])**2

    def learn(self, epoch: int = 10):
        self.train_()
        IOCLoss1, IOCLoss2, lcrLoss, monoLoss, paramLoss = None, None, None, None, None
        for _ in range(epoch):
            IOCLoss1, IOCLoss2, lcrLoss, monoLoss, paramLoss = 0, 0, 0, 0, 0
            # Calculate the loss for preventing parameter decaying
            param_norm = 0
            for param in self.parameters():
                param_norm += torch.norm(param)
            paramLoss = self.decay_coeff * (1 - torch.norm(param_norm))

            # Calculate learned cost loss
            prevC = self.forward(self.sampleE[0].infos[0]['rwinp'].to(self.device))
            for E_trans in self.sampleE:
                for info in E_trans.infos:
                    currC = self.forward(info['rwinp'].to(self.device))
                    IOCLoss1 += currC
                    monoLoss += torch.max(torch.zeros(1).to(self.device), currC - prevC - 1) ** 2
                    prevC = torch.empty_like(currC).copy_(currC)
            IOCLoss1 /= len(self.sampleE)

            # Calculate Max Ent. Loss
            x = torch.zeros(len(self.sampleE+self.sampleL)).double().to(self.device)
            for j, trans_j in enumerate(self.sampleE+self.sampleL):
                temp = 0
                temp -= self.forward(trans_j.infos[0]['rwinp']).to(self.device)+trans_j.infos[0]['log_probs']
                Cp = self.forward(trans_j.infos[0]['rwinp']).to(self.device)
                Cc = self.forward(trans_j.infos[0]['rwinp']).to(self.device)
                Cf = self.forward(trans_j.infos[0]['rwinp']).to(self.device)
                for info in trans_j[1:].infos:
                    Cc.copy_(Cf)
                    Cf = self.forward(info['rwinp']).to(self.device)
                    temp -= Cf+info['log_probs']
                    lcrLoss += (Cf - 2*Cc + Cp) ** 2
                    Cp.copy_(Cc)
                x[j] = temp
            IOCLoss2 = -torch.logsumexp(x, 0)

            IOCLoss = IOCLoss1 + IOCLoss2 + monoLoss + lcrLoss
            self.optimizer.zero_grad()
            IOCLoss.backward()
            self.optimizer.step()
        logger.record("train/Expert_Cost_loss", IOCLoss1.item())
        logger.record("train/Max_Ent._Cost_loss", IOCLoss2.item())
        logger.record("train/Mono_Regularization", monoLoss.item())
        logger.record("train/lcr_Regularization", lcrLoss.item())
        logger.dump()
        return self

    def train_(self):
        self.evalmod = False
        return self.train()

    def eval_(self):
        self.evalmod = True
        return self.eval()
