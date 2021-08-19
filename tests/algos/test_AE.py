import os
import pytest
import pickle

import torch as th
import numpy as np

from algos.torch.AE import VAE

from imitation.data import rollout


@pytest.fixture()
def irl_path():
    return os.path.abspath(os.path.join("..", "..", "IRL"))


def test_AE(irl_path):
    expert_dir = os.path.join(irl_path, "demos", "HPC", "sub01.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    expert_trans = rollout.flatten_trajectories(expert_trajs)
    train_data = th.utils.data.DataLoader(dataset=np.concatenate([expert_trans.obs], axis=1), batch_size=128, shuffle=True)
    vae = VAE(
        inp_dim=6,
        feature_dim=2,
        arch=[64],
    ).double()

    vae.learn(train_loader=train_data, total_epoch=500, weight=2.0)
