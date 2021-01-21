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
                 num_samp: int = 10):
        # The first argument of the arch must be an input size
        super(NNCost, self).__init__()
        self.device = device
        self.optimizer_class = optimizer_class
        self.act_fnc = act_fcn
        self._build(lr, arch)
        self.evalmod = False
        self.num_expert = num_expert
        self.num_samp = num_samp

    def _build(self, lr, arch):
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            layers.append(self.act_fnc())
        layers.append(nn.Linear(arch[-1], 1))
        self.layers = nn.Sequential(*layers)
        self.aparam = nn.Linear(1, 1, bias=False)
        self.optimizer = self.optimizer_class(self.parameters(), lr)

    def sample_trajectory_sets(self, learner_trans, expert_trans):
        self.sampleE = random.sample(expert_trans, self.num_expert)
        self.sampleL = random.sample(learner_trans, self.num_samp)

    def forward(self, obs):
        if self.evalmod:
            with torch.no_grad():
                return self.layers(obs[:-1])**2 + self.aparam(obs[-1:])**2
        else:
            return self.layers(obs[:-1])**2 + self.aparam(obs[-1:])**2

    def learn(self, epoch: int):
        self._train()
        for _ in range(epoch):
            IOCLoss1, IOCLoss2 = 0, 0
            # Calculate learned cost loss
            for E_trans in self.sampleE:
                for i in range(len(E_trans)):
                    IOCLoss1 += self.forward(E_trans[i]['infos']['rwinp'])
            IOCLoss1 /= len(self.sampleE)
            # Calculate Max Ent. Loss
            wjr, wjp = [], []
            with torch.no_grad():
                for trans_k in self.sampleE + self.sampleL:
                    r, p = 0, 0
                    for t in range(len(trans_k)):
                        r -= self.forward(trans_k[t]['infos']['rwinp'])
                        p += trans_k[t]['infos']['log_probs']
                    wjr.append(r)
                    wjp.append(p)
            for j in range(len(self.sampleE+self.sampleL)):
                trans_j = (self.sampleE+self.sampleL)[j]
                cost, Zwjs = 0, 0
                for k in range(len(self.sampleE+self.sampleL)):
                    Zwjs += torch.exp(wjr[k] - wjr[j] - wjp[k] + wjp[j])
                for t in range(len(trans_j)):
                    cost += self.forward(trans_j[t]['infos']['rwinp'])
                IOCLoss2 -= cost / Zwjs
            IOCLoss = IOCLoss1 + IOCLoss2
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