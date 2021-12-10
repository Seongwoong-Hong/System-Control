import os
import pickle
import torch as th
import numpy as np

from algos.torch.ppo import PPO
from common.util import make_env
from common.verification import verify_policy

from matplotlib import pyplot as plt
import matplotlib.lines as lines

irl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def cal_feature_reward(obs, reward_fn):
    mu = []
    weight = reward_fn.layers[-1].weight.detach()
    bias = reward_fn.layers[-1].bias.detach()
    gamma = th.FloatTensor([0.8 ** i for i in range(len(obs[0]))])
    for ob in obs:
        ft = feature_fn(th.from_numpy(ob))
        for layer in reward_fn.layers[:-1]:
            ft = layer.forward(ft)
        mu.append([(ft[:, i] * gamma).sum().item() for i in range(ft.shape[1])])
    mu = th.FloatTensor(mu)
    r = th.mm(mu, weight.t()) + (gamma * bias).sum()
    return mu[:, 0], mu[:, 1], r


def draw_feature_reward():
    env_type = "1DTarget"
    algo_type = "MaxEntIRL"
    env_id = f"{env_type}_disc"
    subj = "viter_disc"
    env = make_env(f"{env_id}-v2")
    # env = make_env(f"{env_type}-v0", wrapper=wrapper, pltqs=pltqs, init_states=init_states)
    name = f"{env_id}/{algo_type}/ext_{subj}_linear_svm_reset"
    with open(f"../demos/{env_type}/{subj}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    lnum = len(expert_trajs)
    for i in range(1, 20):
        fig = plt.figure()
        assert os.path.isfile(model_dir + f"/{i:03d}/agent.pkl")
        with open(os.path.join(model_dir, f"{i:03d}", "reward_net.pkl"), "rb") as f:
            reward_fn = pickle.load(f).to('cpu')
        expt_obs = [expert_trajs[i].obs[:-1] for i in range(lnum)]
        expt_act = [expert_trajs[i].acts for i in range(lnum)]
        if reward_fn.use_action_as_inp:
            expt_obs = np.concatenate([np.array(expt_obs), np.array(expt_act)], axis=2)
        mu1, mu2, r = cal_feature_reward(expt_obs, reward_fn)
        mini1, maxi1 = mu1.min().item(), mu1.max().item()
        mini2, maxi2 = mu2.min().item(), mu2.max().item()
        ax = fig.add_subplot()
        ax.scatter(mu1, mu2, marker='^', color='r')
        with open(model_dir+f"/{i-1:03d}/agent.pkl", "rb") as f:
            algo = pickle.load(f)
        # algo = PPO.load(model_dir + f"/{i - 1:03d}/agent")
        agent_acts, agent_obs, _ = verify_policy(env, algo, deterministic=False, render="None", repeat_num=lnum)
        if reward_fn.use_action_as_inp:
            agent_obs = np.concatenate([np.array(agent_obs)[:, :-1, :], np.array(agent_acts)], axis=2)
        mu1, mu2, r = cal_feature_reward(agent_obs, reward_fn)
        mini1, maxi1 = min([mini1, mu1.min().item()]), max([maxi1, mu1.max().item()])
        mini2, maxi2 = min([mini2, mu2.min().item()]), max([maxi2, mu2.max().item()])
        ax.scatter(mu1, mu2, marker='o', color='c')
        with open(model_dir+f"/{i:03d}/agent.pkl", "rb") as f:
            algo = pickle.load(f)
        # algo = PPO.load(model_dir + f"/{i:03d}/agent")
        agent_acts, agent_obs, _ = verify_policy(env, algo, deterministic=False, render="None", repeat_num=lnum)
        if reward_fn.use_action_as_inp:
            agent_obs = np.concatenate([np.array(agent_obs)[:, :-1, :], np.array(agent_acts)], axis=2)
        mu1, mu2, r = cal_feature_reward(agent_obs, reward_fn)
        mini1, maxi1 = min([mini1, mu1.min().item()]), max([maxi1, mu1.max().item()])
        mini2, maxi2 = min([mini2, mu2.min().item()]), max([maxi2, mu2.max().item()])
        x = [mini1, maxi1, mini2, maxi2]
        ax.scatter(mu1, mu2, marker='o', color='b')
        ax.set_xlim([mini1, maxi1])
        ax.set_ylim([mini2, maxi2])
        w = reward_fn.layers[-1].weight.detach()[0]
        b = reward_fn.layers[-1].bias.detach()[0]
        y = (w[0].item() * np.array(x) + b.item()) / (-w[1].item())
        ax.plot(x[:2], y[:2], color='k', linewidth=4.0)
    plt.show()


def draw_reward_weights():
    log_dir = os.path.join(irl_path, "tmp", "log", "ray_result")
    # get reward_weight and stack
    weigths_stack = []
    for subj in ["sub03", "sub07"]:
        sub_stack = []
        for pert in [1, 2, 3]:
            for trial in [1, 2, 3, 4, 5]:
                name = f"/DiscretizedHuman_{subj}/{subj}_{pert}_{trial}/model/000"
                with open(log_dir + name + "/reward_net.pkl", "rb") as f:
                    rwfn = pickle.load(f)
                sub_stack.append(rwfn.layers[0].weight.detach().numpy().flatten())
        weigths_stack.append(sub_stack)
    weigths_stack = np.array(weigths_stack)
    subplot_name = [f"Feat_{i}" for i in range(8)]
    x = [f"Pert_{i // 5 + 1}" for i in range(15)]
    # x = np.repeat([f"f{i}" for i in range(5, 9)], 5)
    fig = plt.figure(figsize=[36, 12], dpi=300.0)
    for i in range(len(subplot_name)):
        ax = fig.add_subplot(2, 4, i + 1)
        for j in range(len(subplot_name)):
            # ax.scatter(x, weigths_stack[i, j*5:(j+1)*5, 4])
            ax.scatter(x, weigths_stack[0, :, i], color=(0.3, 0.3, 0.8))
            ax.scatter(x, weigths_stack[1, :, i], color=(0.8, 0.3, 0.3))
        ax.legend(["sub03", "sub07"], ncol=1, columnspacing=0.1, fontsize=15)
        ax.set_xlabel("Perturbation", labelpad=15.0, fontsize=28)
        ax.set_ylabel("weight", labelpad=15.0, fontsize=28)
        ax.set_title(subplot_name[i], fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=15)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    def feature_fn(x):
        # return x
        return th.cat([x, x ** 2], dim=1)
        # return th.cat([x, x**2, x**3, x**4], dim=1)


    # draw_feature_reward()
    draw_reward_weights()
