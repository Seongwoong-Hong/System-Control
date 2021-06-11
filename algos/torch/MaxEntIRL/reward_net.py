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
        self.in_features = inp
        self._build(lr, [inp] + arch)
        self.trainmode = False

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
        self.layers = nn.Sequential(*layers)
        self.optimizer = self.optim_cls(self.parameters(), lr)

    def forward(self, x):
        x = self.feature_fn(x).to(self.device)
        if self.trainmode:
            return self.layers(x)
        else:
            with th.no_grad():
                return self.layers(x)

    def train(self, mode=True):
        self.trainmode = mode
        return super(RewardNet, self).train(mode=mode)

    def eval(self):
        return self.train(False)


class CNNRewardNet(RewardNet):
    def _build(self, lr, arch):
        arch[0] = 1
        layers = []
        if self.act_fnc is not None:
            for i in range(len(arch) - 1):
                layers.append(nn.Conv1d(in_channels=arch[i], out_channels=arch[i+1], kernel_size=3, padding=1))
                layers.append(self.act_fnc())
        else:
            for i in range(len(arch) - 1):
                layers.append(nn.Conv1d(in_channels=arch[i], out_channels=arch[i+1], kernel_size=3, padding=1))
        self.conv_layers = nn.Sequential(*layers)
        self.fcnn = nn.Linear(arch[-1] * self.in_features, 1, bias=False)
        self.optimizer = self.optim_cls(self.parameters(), lr)

    def forward(self, x):
        x = self.feature_fn(x)
        x = x.unsqueeze(1).to(self.device)
        if self.trainmode:
            x = self.conv_layers(x)
            x = x.view(-1, self.fcnn.in_features)
            return self.fcnn(x)
        else:
            with th.no_grad():
                x = self.conv_layers(x)
                x = x.view(-1, self.fcnn.in_features)
                return self.fcnn(x)
