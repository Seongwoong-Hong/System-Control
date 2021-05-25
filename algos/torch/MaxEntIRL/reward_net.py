import torch as th

from torch import nn
from typing import List


class RewardNet(nn.Module):
    def __init__(
            self,
            inp,
            arch: List[int],
            lr: float = 1e-3,
            device: str = 'cuda',
            optim_cls=th.optim.Adam,
            activation_fn=th.nn.Tanh,
            feature_fn=lambda x: x,
    ):
        super(RewardNet, self).__init__()
        self.device = device
        self.act_fnc = activation_fn
        self.feature_fn = feature_fn
        self.optim_cls = optim_cls
        self._build(lr, [inp] + arch)
        self.evalmod = False

    def _build(self, lr, arch):
        layers = []
        if self.act_fnc is not None:
            for i in range(len(arch) - 1):
                layers.append(nn.Linear(arch[i], arch[i + 1]))
                layers.append(self.act_fnc())
        else:
            for i in range(len(arch) - 1):
                layers.append(nn.Linear(arch[i], arch[i + 1]))
        layers.append(nn.Linear(arch[-1], 1, bias=False))
        self.layers = nn.Sequential(*layers).to(self.device)
        self.optimizer = self.optim_cls(self.parameters(), lr)

    def forward(self, x):
        # x = self.feature_fn(x)
        if self.evalmod:
            with th.no_grad():
                return self.layers(x.to(self.device))
        else:
            return self.layers(x.to(self.device))

    def train(self, mode=True):
        self.evalmod = False
        return super(RewardNet, self).train(mode=mode)

    def eval(self):
        self.evalmod = True
        return super(RewardNet, self).eval()
