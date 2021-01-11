import torch, random
from torch import nn

class RewfromMat(nn.Module):
    def __init__(self,
                 inp,
                 device='cpu',
                 optimizer_class=torch.optim.Adam,
                 lr=1e-4):
        super(RewfromMat, self).__init__()
        self.device = device
        self.layer1 = nn.Linear(inp, 2*inp)
        self.layer2 = nn.Linear(2*inp, 1)
        self.relu = nn.ReLU()
        self.optimizer_class = optimizer_class
        self._build(lr)

    def _build(self, lr):
        self.optimizer = self.optimizer_class(self.parameters(), lr)

    def sample_trajectory_sets(self, learner_trans, expert_trans):
        self.sampleL = random.sample(learner_trans, 10)
        self.sampleE = random.sample(expert_trans, 5)

    def forward(self, obs):
        out = self.layer1(obs)
        return self.layer2(out)

    def learn(self, epoch):
        self.train()
        for _ in range(epoch):
            IOCLoss = 0.0
            for E_trans in self.sampleE:
                for i in range(len(E_trans)):
                    IOCLoss -= self.forward(torch.from_numpy(E_trans[i]['obs']).to(self.device))
            IOCLoss /= len(self.sampleE)
            for trans_j in self.sampleE+self.sampleL:
                cost, wjZ = 0, 0
                for t in range(len(trans_j)):
                    cost -= self.forward(torch.from_numpy(trans_j[t]['obs']).to(self.device))
                for trans_k in self.sampleE+self.sampleL:
                    temp = 0
                    for t in range(len(trans_k)):
                        with torch.no_grad():
                            temp += -self.forward(torch.from_numpy(trans_k[t]['obs']).to(self.device)) \
                                    + self.forward(torch.from_numpy(trans_j[t]['obs']).to(self.device)) \
                                    - trans_k[t]['infos']['log_probs'] + trans_j[t]['infos']['log_probs']
                    wjZ += temp
                IOCLoss -= cost / wjZ
            self.optimizer.zero_grad()
            IOCLoss.backward()
            self.optimizer.step()
        print("Loss: {:.2f}".format(IOCLoss.item()))
        return self