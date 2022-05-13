import os
import pytest
import torch as th
import numpy as np
import pickle
from common.util import make_env
from imitation.data.rollout import flatten_trajectories
import matplotlib.pyplot as plt


def test_svm():
    with open("../../IRL/demos/1DTarget/viter_disc.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
        expt_trans = flatten_trajectories(expt_trajs)
    with open("../../IRL/demos/1DTarget/viter_disc0.pkl", "rb") as f:
        agent_trajs = pickle.load(f)
        agent_trans = flatten_trajectories(agent_trajs)
    with open("../../IRL/tmp/log/1DTarget_disc/MaxEntIRL/ext_viter_disc_linear_svm_reset/model/000/reward_net.pkl", "rb") as f:
        reward_fn = pickle.load(f)
    with open("../../IRL/tmp/log/1DTarget_disc/MaxEntIRL/ext_viter_disc_linear_svm_reset/model/000/agent.pkl", "rb") as f:
        agent = pickle.load(f)
    agent_input = feature_fn(th.from_numpy(np.concatenate([agent_trans.obs], axis=1)))
    expt_input = feature_fn(th.from_numpy(np.concatenate([expt_trans.obs], axis=1)))
    agent_gammas = th.FloatTensor([0.8 ** (i % 20) for i in range(len(agent_trans))])
    expt_gammas = th.FloatTensor([0.8 ** (i % 20) for i in range(len(expt_trans))])

    agent_mu, expt_mu = th.zeros([len(agent_trajs), 2]), th.zeros([len(expt_trajs), 2])
    for i in range(len(agent_trajs)):
        agent_mu[i, 0] = (agent_gammas * agent_input[:, 0].flatten())[i * 20:(i + 1) * 20].sum()
        agent_mu[i, 1] = (agent_gammas * agent_input[:, 1].flatten())[i * 20:(i + 1) * 20].sum()
        # agent_mu[i, 2] = (agent_gammas * agent_input[:, 2].flatten())[i * 1:(i + 1) * 1].sum()
        # agent_mu[i, 3] = (agent_gammas * agent_input[:, 3].flatten())[i * 1:(i + 1) * 1].sum()
    for i in range(len(expt_trajs)):
        expt_mu[i, 0] = (expt_gammas * expt_input[:, 0].flatten())[i * 20:(i + 1) * 20].sum()
        expt_mu[i, 1] = (expt_gammas * expt_input[:, 1].flatten())[i * 20:(i + 1) * 20].sum()
        # expt_mu[i, 2] = (expt_gammas * expt_input[:, 2].flatten())[i * 1:(i + 1) * 1].sum()
        # expt_mu[i, 3] = (expt_gammas * expt_input[:, 3].flatten())[i * 1:(i + 1) * 1].sum()

    w1, w2 = reward_fn.layers[0].weight.detach().flatten()
    w1, w2 = w1.item(), w2.item()
    b = reward_fn.layers[0].bias.detach().item()

    a_r = w1 * agent_mu[:11, 0] + w2 * agent_mu[:11, 1] + b
    e_r = w1 * expt_mu[:11, 0] + w2 * expt_mu[:11, 1] + b

    print('end')


if __name__ == "__main__":
    def feature_fn(x):
        return th.cat([x, x**2], dim=1)

    test_svm()