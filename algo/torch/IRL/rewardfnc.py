import torch, random, time
from torch import nn

class RewfromMat(nn.Module):
    def __init__(self,
                 inp,
                 device='cpu',
                 optimizer_class=torch.optim.Adam,
                 lr=3e-5):
        super(RewfromMat, self).__init__()
        self.device = device
        self.layer1 = nn.Linear(inp, inp)
        self.optimizer_class = optimizer_class
        self._build(lr)
        self.evalmod = False

    def _build(self, lr):
        self.optimizer = self.optimizer_class(self.parameters(), lr)

    def sample_trajectory_sets(self, learner_trans, expert_trans):
        self.sampleL = random.sample(learner_trans, 10)
        self.sampleE = random.sample(expert_trans, 10)

    def forward(self, obs):
        if self.evalmod:
            with torch.no_grad():
                out = self.layer1(obs)
                return -out @ out.T
        else:
            out = self.layer1(obs)
            return -out @ out.T

    def learn(self, epoch):
        self._train()
        for _ in range(epoch):
            IOCLoss1, IOCLoss2 = 0, 0
            # Calculate learned cost loss
            for E_trans in self.sampleE:
                for i in range(len(E_trans)):
                    IOCLoss1 -= self.forward(E_trans[i]['infos']['rwinp'])
            IOCLoss1 /= len(self.sampleE)
            # Calculate Max Ent. Loss
            wjr, wjp = [], []
            with torch.no_grad():
                for trans_k in self.sampleE + self.sampleL:
                    r, p = 0, 0
                    for t in range(len(trans_k)):
                        r += self.forward(trans_k[t]['infos']['rwinp'])
                        p += trans_k[t]['infos']['log_probs']
                    wjr.append(r)
                    wjp.append(p)
            for j in range(len(self.sampleE+self.sampleL)):
                trans_j = (self.sampleE+self.sampleL)[j]
                cost, Zwjs = 0, 0
                for k in range(len(self.sampleE+self.sampleL)):
                    Zwjs += torch.exp(wjr[k] - wjr[j]) / torch.exp(wjp[k] - wjp[j])
                for t in range(len(trans_j)):
                    cost -= self.forward(trans_j[t]['infos']['rwinp'])
                IOCLoss2 -= cost / Zwjs
            # IOCLoss = IOCLoss1 + IOCLoss2
            self.optimizer.zero_grad()
            IOCLoss1.backward()
            self.optimizer.step()
        print("Loss for Expert cost: {:.2f}, Loss for Max Ent.: {:.2f}".format(IOCLoss1.item(), IOCLoss2.item()))
        return self

    def _train(self):
        self.evalmod = False
        return self.train()

    def _eval(self):
        self.evalmod = True
        return self.eval()