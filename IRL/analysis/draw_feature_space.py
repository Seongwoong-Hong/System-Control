import os
import pickle
import torch as th
import numpy as np

from algos.torch.ppo import PPO
from common.util import make_env
from common.verification import verify_policy

from matplotlib import pyplot as plt
import matplotlib.lines as lines


def cal_feature_reward(obs, reward_fn):
    mu = []
    weight = reward_fn.layers[-1].weight.detach()
    bias = reward_fn.layers[-1].bias.detach()
    gamma = th.DoubleTensor([0.99 ** i for i in range(len(obs[0]))])
    for ob in obs:
        ft = feature_fn(th.from_numpy(ob))
        for layer in reward_fn.layers[:-1]:
            ft = layer.forward(ft)
        mu.append([(ft[:, i] * gamma).sum().item() for i in range(ft.shape[1])])
    mu = th.DoubleTensor(mu)
    r = th.mm(mu, weight.t()) + bias
    return mu[:, 0], mu[:, 1], r


def draw_feature_reward():
    env_type = "1DTarget"
    algo_type = "MaxEntIRL"
    env_id = f"{env_type}_disc"
    subj = "ppo_disc"
    env = make_env(f"{env_id}-v2")
    # env = make_env(f"{env_type}-v0", wrapper=wrapper, pltqs=pltqs, init_states=init_states)
    name = f"{env_id}/{algo_type}/no_{subj}_samp_ppoagent_svm_reset"
    with open(f"../demos/{env_type}/{subj}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    lnum = len(expert_trajs)
    expt_obs = [expert_trajs[i].obs[:-1] for i in range(lnum)]
    expt_act = [expert_trajs[i].acts for i in range(lnum)]
    for i in range(1, 3):
        assert os.path.isfile(model_dir + f"/{i:03d}/agent.zip")
        with open(os.path.join(model_dir, f"{i:03d}", "reward_net.pkl"), "rb") as f:
            reward_fn = pickle.load(f).to('cpu')
        if reward_fn.use_action_as_inp:
            expt_obs = np.concatenate([np.array(expt_obs), np.array(expt_act)], axis=2)
        mu1, mu2, r = cal_feature_reward(expt_obs, reward_fn)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(mu1, mu2, marker='^', color='r')
        w = reward_fn.layers[-1].weight.detach()[0]
        b = reward_fn.layers[-1].bias.detach()[0]
        x = th.tensor([-5, 1])
        y = (w[0] * x + b) / (-w[1])
        ax.add_artist(lines.Line2D(x, y, color='k', linewidth=4.0))
        algo = PPO.load(model_dir + f"/{i - 1:03d}/agent")
        agent_acts, agent_obs, _ = verify_policy(env, algo, deterministic=False, render="None", repeat_num=lnum)
        if reward_fn.use_action_as_inp:
            agent_obs = np.concatenate([np.array(agent_obs)[:, :-1, :], np.array(agent_acts)], axis=2)
        mu1, mu2, r = cal_feature_reward(agent_obs, reward_fn)
        ax.scatter(mu1, mu2, marker='o', color='c')
        algo = PPO.load(model_dir + f"/{i:03d}/agent")
        agent_acts, agent_obs, _ = verify_policy(env, algo, deterministic=False, render="None", repeat_num=lnum)
        if reward_fn.use_action_as_inp:
            agent_obs = np.concatenate([np.array(agent_obs)[:, :-1, :], np.array(agent_acts)], axis=2)
        mu1, mu2, r = cal_feature_reward(agent_obs, reward_fn)
        ax.scatter(mu1, mu2, marker='o', color='b')
        plt.show()


if __name__ == "__main__":
    def feature_fn(x):
        return x
        # return th.cat([x, x**2], dim=1)
        # return th.cat([x, x**2, x**3, x**4], dim=1)
    draw_feature_reward()
