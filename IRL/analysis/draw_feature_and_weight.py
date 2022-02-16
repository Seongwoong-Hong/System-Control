import os
import pickle
import torch as th
import numpy as np

from algos.torch.ppo import PPO
from common.util import make_env, CPU_Unpickler
from common.verification import verify_policy

from matplotlib import pyplot as plt

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
    unnormalize = True
    log_dir = os.path.join(irl_path, "tmp", "log", "DiscretizedHuman", "MaxEntIRL", "sq_normalize_finite_")
    # get reward_weight and stack
    weights_stack = []
    label_name = [f"sub{i:02d}" for i in [6]]
    for subj in label_name:
        weights_per_actuation = []
        for actuation in range(1, 2):
            weights = []
            for trial in range(1, 6):
                name = f"17171719_quadcost/{subj}_{actuation}_{trial}/model"
                with open(log_dir + name + "/reward_net.pkl", "rb") as f:
                    reward_weight = CPU_Unpickler(f).load().layers[0].weight.detach()
                if unnormalize:
                    gains = feature_fn(th.Tensor([[0.16, 0.67, 0.8, 2.4, 100., 100.]]))
                    reward_weight /= gains
                weights.append(reward_weight.numpy().flatten())
            # weights = np.array(weights) / np.linalg.norm(np.array(weights), axis=-1, keepdims=True)  # normalize
            weights = np.array(weights)
            weights_per_actuation.append(
                np.append(weights.mean(axis=0, keepdims=True), weights.std(axis=0, keepdims=True), axis=0)
            )
        weights_stack.append(weights_per_actuation)
    weights_stack = np.array(weights_stack)
    w_mean = weights_stack[:, :, 0, :]
    w_std = weights_stack[:, :, 1, :]

    x1 = [f"{i}cm" for i in [3]]#, 4.5, 6, 7.5, 9, 12]]
    subplot_name = [f"$\omega_{{{i + 1}}}$" for i in range(6)]
    # x = np.repeat([f"f{i}" for i in range(5, 9)], 5)
    for subj_idx, subj in enumerate(label_name):
        weight_fig = plt.figure(figsize=[36, 8], dpi=100.0)
        for weight_idx, weight_name in enumerate(subplot_name):
            ax = weight_fig.add_subplot(1, 6, weight_idx + 1)
            ax.errorbar(x1, w_mean[subj_idx, :, weight_idx], yerr=w_std[subj_idx, :, weight_idx], fmt='o')
            ax.legend([subj], ncol=1, columnspacing=0.1, fontsize=15)
            ax.set_xlabel("perturbation", labelpad=15.0, fontsize=28)
            ax.set_ylabel("weight", labelpad=15.0, fontsize=28)
            # ax.set_ylim(-0.3, .9)
            ax.set_title(weight_name, fontsize=32)
            ax.tick_params(axis='both', which='major', labelsize=15)
            weight_fig.tight_layout()
        # plt.savefig(f"figures/disc_reward_weights/linspace_disc/{subj}_weights_notnorm.png")
        plt.show()


if __name__ == "__main__":
    def feature_fn(x):
        # return x
        return x ** 2
        # return th.cat([x, x ** 2], dim=1)

    # draw_feature_reward()
    draw_reward_weights()
