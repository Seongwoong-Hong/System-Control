from typing import List
import torch as th
from torch import nn


class VAE(nn.Module):
    def __init__(
            self,
            inp_dim: int,
            feature_dim: int,
            arch: List,
            lr: float = 1e-3,
            alpha: float = 0.0,
            device: str = 'cpu',
            optim_cls=th.optim.Adam,
            activation_fn=nn.ReLU,
    ):
        super(VAE, self).__init__()
        self.device = device
        self.optim_cls = optim_cls
        self.act_fn = activation_fn
        self.inp_dim = inp_dim
        self.feature_dim = feature_dim
        self._build([inp_dim] + arch, feature_dim, lr, alpha)

    def _build(self, arch, feature_dim, lr, alpha):
        enc_layers = []
        dec_layers = []
        if self.act_fn is not None:
            for i in range(len(arch) - 1):
                enc_layers.append(nn.Linear(arch[i], arch[i + 1]))
                enc_layers.append(self.act_fn())
                dec_layers.append(self.act_fn())
                dec_layers.append(nn.Linear(arch[-i - 1], arch[-i - 2]))
        else:
            for i in range(len(arch) - 1):
                enc_layers.append(nn.Linear(arch[i], arch[i + 1]))
                dec_layers.append(nn.Linear(arch[-i - 1], arch[-i - 2]))
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_layer = nn.Linear(arch[-1], feature_dim)
        self.var_layer = nn.Linear(arch[-1], feature_dim)
        dec_layers = [nn.Linear(feature_dim, arch[-1])] + dec_layers + [nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)
        self.optimizer = self.optim_cls(self.parameters(), lr, weight_decay=alpha)

    def sampling(self, mu, log_var):
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.mu_layer(self.encoder(x.view(-1, self.inp_dim)))
        log_var = self.var_layer(self.encoder(x.view(-1, self.inp_dim)))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss_fn(self, recon_x, x, mu, log_var, weight):
        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, self.inp_dim), reduction='sum')
        KLD = weight * 0.5 * th.sum(1 + log_var - mu ** 2 - log_var.exp())
        return BCE, KLD

    def learn(self, train_loader, total_epoch, weight):
        self.train()
        bce_loss = 0.0
        kld_loss = 0.0
        for epoch in range(total_epoch):
            for batch_idx, data in enumerate(train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, log_var = self.forward(data)
                BCE, KLD = self.loss_fn(recon_batch, data, mu, log_var, weight)
                loss = BCE - KLD
                loss.backward()
                bce_loss += BCE.item()
                kld_loss += KLD.item()
                self.optimizer.step()
            print(f'====> Epoch: {epoch} BCE loss: {bce_loss/len(train_loader.dataset):.4f}\tKLD_loss: {kld_loss/len(train_loader.dataset):.4f}')
