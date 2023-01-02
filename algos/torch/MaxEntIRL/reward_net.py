import os
import pickle
import torch as th
from copy import deepcopy

from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from typing import List


class RewardNet(nn.Module):
    def __init__(
            self,
            inp: int,
            arch: List[int],
            feature_fn,
            use_action_as_inp: bool,
            device: str = 'cpu',
            optim_cls=th.optim.Adam,
            optim_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            activation_fn=th.nn.Tanh,
    ):
        super(RewardNet, self).__init__()
        self.use_action_as_inp = use_action_as_inp
        self._device = device
        self.act_fnc = activation_fn
        self.feature_fn = feature_fn
        self.optim_cls = optim_cls
        self.optim_kwargs = optim_kwargs
        if self.optim_kwargs is None:
            self.optim_kwargs = {}
        self.lr_scheduler = None
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.in_features = inp
        self._arch = [self.in_features] + arch
        self._build()
        self.trainmode = True

    def _build(self):
        layers = []
        if self.act_fnc is not None:
            for i in range(len(self._arch) - 1):
                layers.append(nn.Linear(self._arch[i], self._arch[i + 1]))
                layers.append(self.act_fnc())
        else:
            for i in range(len(self._arch) - 1):
                layers.append(nn.Linear(self._arch[i], self._arch[i + 1]))
        layers.append(nn.Linear(self._arch[-1], 1, bias=False))
        self.layers = nn.Sequential(*layers)
        # self.layers[0].weight = th.nn.Parameter(
        #     th.tensor([0.686, 0.686, 0.072, 0.072, -2.401, -2.401, -0.036, -0.036]))
        self.optimizer = self.optim_cls(self.parameters(), **self.optim_kwargs)
        if self.lr_scheduler_cls:
            self.lr_scheduler = self.lr_scheduler_cls(self.optimizer, **self.lr_scheduler_kwargs)
        self.to(self._device)

    def reset_optim(self):
        self.optimizer = self.optim_cls(self.parameters(), **self.optim_kwargs)
        self.to(self._device)

    def reset(self):
        self._build()

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.feature_fn(x)
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

    def save(self, log_dir):
        feature_fn = deepcopy(self.feature_fn)
        self.feature_fn = None
        with open(log_dir + ".tmp", "wb") as f:
            pickle.dump(self.to('cpu'), f)
        os.replace(log_dir + ".tmp", log_dir + ".pkl")
        self.to(self._device)
        self.feature_fn = feature_fn

    @property
    def device(self):
        if str(next(self.parameters()).device) != self._device:
            self._device = str(next(self.parameters()).device)
        return self._device


class CNNRewardNet(RewardNet):
    def _build(self):
        self._arch[0] = 1
        self.len_act_w = len(self.feature_fn(th.zeros(2)))
        layers = []
        for i in range(len(self._arch) - 1):
            layers.append(nn.Conv1d(in_channels=self._arch[i], out_channels=self._arch[i + 1], kernel_size=(3,), padding=1))
            if self.act_fnc is not None:
                layers.append(self.act_fnc())
        layers.append(nn.AvgPool1d(4))
        self.feature_layers = nn.Sequential(*layers)
        self.reward_layer = nn.Linear(self._arch[-1] + self.len_act_w, 1, bias=False)
        self.optimizer = self.optim_cls(self.parameters(), **self.optim_kwargs)
        if self.lr_scheduler_cls:
            self.lr_scheduler = self.lr_scheduler_cls(self.optimizer, **self.lr_scheduler_kwargs)
        self.to(self._device)

    def forward(self, x: th.Tensor) -> th.Tensor:
        u = self.feature_fn(x[:, -2:])
        x_f = self.feature_layers(self.feature_fn(x[:, :-2])[:, None, :]).reshape(-1, self._arch[-1])
        if self.trainmode:
            return -(th.sum(x_f * self.reward_layer.weight[:, :-self.len_act_w].square(), dim=-1)
                     + th.sum(u * self.reward_layer.weight[:, -self.len_act_w:].square(), dim=-1))
        else:
            with th.no_grad():
                return -(th.sum(x_f * self.reward_layer.weight[:, :-self.len_act_w].square(), dim=-1)
                         + th.sum(u * self.reward_layer.weight[:, -self.len_act_w:].square(), dim=-1))


class QuadraticRewardNet(RewardNet):
    def _build(self):
        self.len_act_w = 2
        self._arch[0] -= 1
        feature_layers = []
        for i in range(len(self._arch) - 1):
            feature_layers.append(nn.Linear(self._arch[i], self._arch[i + 1]))
            if self.act_fnc is not None:
                feature_layers.append(self.act_fnc())
        self.feature_layers = nn.Sequential(*feature_layers)
        self.reward_layer = nn.Linear(in_features=self._arch[-1] + 1, out_features=1, bias=False)
        self.optimizer = self.optim_cls(self.parameters(), **self.optim_kwargs)
        if self.lr_scheduler_cls:
            self.lr_scheduler = self.lr_scheduler_cls(self.optimizer, **self.lr_scheduler_kwargs)
        self.to(self._device)

    def forward(self, x: th.Tensor) -> th.Tensor:
        u = self.feature_fn(x[:, -1:])
        x_f = self.feature_layers(self.feature_fn(x[:, :-1]))
        if self.trainmode:
            return -(th.sum(x_f * self.reward_layer.weight[:, :-self.len_act_w].square(), dim=-1)
                     + th.sum(u * self.reward_layer.weight[:, -self.len_act_w:].square(), dim=-1))
        else:
            with th.no_grad():
                return -(th.sum(x_f * self.reward_layer.weight[:, :-self.len_act_w].square(), dim=-1)
                         + th.sum(u * self.reward_layer.weight[:, -self.len_act_w:].square(), dim=-1))


class XXRewardNet(QuadraticRewardNet):
    def forward(self, x: th.Tensor) -> th.Tensor:
        x_f, u = th.split(self.feature_fn(x), 4, dim=1)
        if self.trainmode:
            return -(th.sum(x_f * self.reward_layer.weight[:, :-2*self.len_act_w].square(), dim=-1)
                     + th.sum(u * self.reward_layer.weight[:, -2*self.len_act_w:].square(), dim=-1))
        else:
            with th.no_grad():
                return -(th.sum(x_f * self.reward_layer.weight[:, :-2*self.len_act_w].square(), dim=-1)
                         + th.sum(u * self.reward_layer.weight[:, -2*self.len_act_w:].square(), dim=-1))


class LURewardNet(RewardNet):
    def _build(self):
        self.w_th = th.nn.Parameter(th.rand(10))
        self.w_tq = th.nn.Parameter(th.rand(3))
        self.optimizer = self.optim_cls(self.parameters(), **self.optim_kwargs)
        self.lr_scheduler = None
        if self.lr_scheduler_cls:
            self.lr_scheduler = self.lr_scheduler_cls(self.optimizer, **self.lr_scheduler_kwargs)
        self.to(self._device)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x = self.feature_fn(x)
        qu, ru = th.zeros(4, 4), th.zeros(2, 2)
        qu[0, :] = self.w_th[:4]
        qu[1, 1:] = self.w_th[4:7]
        qu[2, 2:] = self.w_th[7:9]
        qu[3, 3] = self.w_th[9]
        ru[0, :] = self.w_tq[:2]
        ru[1, 1] = self.w_tq[2]
        q = qu.t() @ qu
        r = ru.t() @ ru
        if self.trainmode:
            return - th.sum((x[:, :4] @ q) * x[:, :4], dim=-1) - th.sum((x[:, -2:] @ r) * x[:, -2:], dim=-1)
        else:
            with th.no_grad():
                return - th.sum((x[:, :4] @ q) * x[:, :4], dim=-1) - th.sum((x[:, -2:] @ r) * x[:, -2:], dim=-1)
