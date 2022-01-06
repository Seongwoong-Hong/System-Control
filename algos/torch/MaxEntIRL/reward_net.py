import os
import pickle
import torch as th

from torch import nn
from typing import List


class RewardNet(nn.Module):
    def __init__(
            self,
            inp: int,
            arch: List[int],
            feature_fn,
            use_action_as_inp: bool,
            scale: float = 1.0,
            lr: float = 1e-3,
            norm_coeff: float = 0.0,
            device: str = 'cpu',
            optim_cls=th.optim.Adam,
            activation_fn=th.nn.Tanh,
    ):
        super(RewardNet, self).__init__()
        self.use_action_as_inp = use_action_as_inp
        self.scale = scale
        self._device = device
        self.act_fnc = activation_fn
        self.feature_fn = feature_fn
        self.optim_cls = optim_cls
        self.in_features = inp
        self._build_args = [lr, norm_coeff, [self.in_features] + arch]
        self._build(lr, norm_coeff, [self.in_features] + arch)
        self.trainmode = True

    def _build(self, lr: float, norm_coeff: float, arch: List[int]):
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
        self.optimizer = self.optim_cls(self.parameters(), lr, weight_decay=norm_coeff)

    def reset_optim(self):
        self.optimizer = self.optim_cls(self.parameters(), self._build_args[0], weight_decay=self._build_args[1])
        self.to(self.device)

    def reset(self):
        self._build(*self._build_args)
        self.to(self.device)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.feature_fn(x).to(self.device)
        if self.trainmode:
            return self.scale * self.layers(x)
        else:
            with th.no_grad():
                return self.scale * self.layers(x)

    def train(self, mode=True):
        self.trainmode = mode
        return super(RewardNet, self).train(mode=mode)

    def eval(self):
        return self.train(False)

    def save(self, log_dir):
        state = self.__dict__.copy()
        del state['feature_fn']
        self.__dict__.update(state)
        self.feature_fn = None
        with open(log_dir + ".tmp", "wb") as f:
            pickle.dump(self.cpu(), f)
        self.to(self._device)
        os.replace(log_dir + ".tmp", log_dir + ".pkl")

    @property
    def device(self):
        if str(next(self.parameters()).device) != self._device:
            self._device = str(next(self.parameters()).device)
        return self._device


class CNNRewardNet(RewardNet):
    def _build(self, lr, norm_coeff, arch):
        arch[0] = 1
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Conv1d(in_channels=arch[i], out_channels=arch[i + 1], kernel_size=(3,), padding=1))
            if self.act_fnc is not None:
                layers.append(self.act_fnc())
        self.conv_layers = nn.Sequential(*layers)
        self.fcnn = nn.Linear(arch[-1] * self.in_features, 1, bias=False)
        self.optimizer = self.optim_cls(self.parameters(), lr, weight_decay=norm_coeff)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.feature_fn(x)
        x = x.unsqueeze(1)
        if self.trainmode:
            x = self.conv_layers(x)
            x = x.view(-1, self.fcnn.in_features)
            return self.scale * self.fcnn(x)
        else:
            with th.no_grad():
                x = self.conv_layers(x)
                x = x.view(-1, self.fcnn.in_features)
                return self.scale * self.fcnn(x)
