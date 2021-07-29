import os
import pickle
import numpy as np
import torch as th

from copy import deepcopy
from scipy import io
from matplotlib import cm
from matplotlib import pyplot as plt
from imitation.data.rollout import flatten_trajectories, make_sample_until, generate_trajectories

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import verify_policy
from IRL.scripts.project_policies import def_policy


def draw_trajectories():
    env_type = "IDP_custom"
    env = make_env(f"{env_type}-v0", use_vec_env=False)
    name = f"{env_type}/BC/extcnn_lqr_ppo_deep_noreset_rewfirst"
    model_dir = os.path.join("..", "tmp", "log", name, "model", "018")
    algo = SAC.load(model_dir + "/agent")
    # expt = def_policy("IDP", env)
    expt = PPO.load("../../RL/IDP/tmp/log/IDP_custom/ppo/policies_10/ppo0")
    # i = int(7.5e6)
    # expt = PPO.load(os.path.join("..", "..", "RL", "IDP", "tmp", "log", env_type, "ppo", "policies_10", f"{i:012d}", "model.pkl"))
    _, agent_obs, _ = verify_policy(env, algo, deterministic=True, render="rgb_array", repeat_num=1)
    env = make_env(f"{env_type}-v0", use_vec_env=False)
    _, expt_obs, _ = verify_policy(env, expt, deterministic=True, render="rgb_array", repeat_num=1)
    t = np.linspace(0, env.dt*600, 601)
    for j in range(1):
        errors = 100 * (agent_obs[j] - expt_obs[j]) / expt_obs[j]
        yval_list = [agent_obs[j], expt_obs[j]]
        fig = plt.figure(figsize=[7.2, 6.4], dpi=300.0)
        plt.plot(t, yval_list[0][:, 0], color=(19 / 255, 0 / 255, 182 / 255, 1), lw=3)
        plt.plot(t, yval_list[1][:, 0], color=(19 / 255, 0 / 255, 182 / 255, 0.4), lw=3)
        plt.plot(t, yval_list[0][:, 1], color=(255 / 255, 105 / 255, 21 / 255, 1), lw=3)
        plt.plot(t, yval_list[1][:, 1], color=(255 / 255, 105 / 255, 21 / 255, 0.6), lw=3)
        plt.legend(['', '', 'learned', 'original'], ncol=2, columnspacing=0.1, fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.ylim(-0.1, 0.2)
        plt.xlim(0.0, 5.0)
        plt.axhline(y=0.0, linestyle=':', color='0.5')
        plt.title("Simulation Result", fontsize=28, pad=30)
        plt.xlabel("time", fontsize=24)
        plt.ylabel(r"$\theta$s", fontsize=24)
        # plt.savefig(f"figures/IDP/BC/cost_plot.png")
        plt.show()


def draw_costfigure():
    def expt_cost(inp):
        return inp[0, :2].square().sum() + 0.1 * inp[0, 2:4].square().sum() + 1e-5 * (200 * inp[0, 6:]).square().sum()
    env_type = "IDP"
    env_id = f"{env_type}_custom"
    pltqs = []
    if env_type == "HPC":
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            file = os.path.join("..", "demos", env_type, "sub01", f"sub01i{i + 1}.mat")
            pltqs += [io.loadmat(file)['pltq']]
    env = make_env(f"{env_id}-v0", use_vec_env=False, pltqs=pltqs)
    name = "extcnn_lqr_ppo_deep_noreset_rewfirst"
    num = 18
    load_dir = f"../tmp/log/{env_id}/BC/{name}/model"
    algo = SAC.load(load_dir + f"/{num:03d}/agent")
    # algo = def_policy("IDP", env)
    # algo = PPO.load(f"../../RL/IDP/tmp/log/IDP_custom/ppo/policies_1/ppo0")
    with open(load_dir + f"/{num:03d}/reward_net.pkl", "rb") as f:
        reward_fn = pickle.load(f).double()

    ndim, nact = env.observation_space.shape[0], env.action_space.shape[0]
    d1, d2 = np.meshgrid(np.linspace(-0.3, 0.3, 100), np.linspace(-0.3, 0.3, 100))
    pact = np.zeros((100, 100), dtype=np.float64)
    cost1, cost2 = np.zeros(d1.shape), np.zeros(d1.shape)
    for i in range(d1.shape[0]):
        for j in range(d1.shape[1]):
            iobs = np.zeros(ndim)
            iobs[0], iobs[2] = deepcopy(d1[i][j]), deepcopy(d2[i][j])
            iacts, _ = algo.predict(np.array(iobs), deterministic=True)
            pact[i][j] = iacts[0]
            inp = th.from_numpy(np.append(iobs, iacts)).double().to(algo.device).reshape(1, -1)
            cost1[i][j] = expt_cost(inp)
            cost2[i][j] = -reward_fn(inp).item()
    cost1 = (cost1 - np.min(cost1)) / (np.max(cost1) - np.min(cost1))
    cost2 = (cost2 - np.min(cost2)) / (np.max(cost2) - np.min(cost2))
    title_list = ["original cost", "learned cost"]
    yval_list = [cost1, cost2]
    xlabel, ylabel = r"$\theta_1$", r"$\omega_1$"
    max_list = [1.0, 1.0]
    min_list = [0.0, 0.0]
    fig = plt.figure(figsize=[12, 5.8], dpi=300.0)
    for i in range(2):
        ax = fig.add_subplot(1, 2, (i + 1), projection='3d')
        # d1, d2 = np.meshgrid(d1, d2)
        surf = ax.plot_surface(d1, d2, yval_list[i], rstride=1, cstride=1, cmap=cm.rainbow,
                               vmax=max_list[i], vmin=min_list[i])
        # clb = fig.colorbar(surf, ax=ax)
        ax.set_xlabel(xlabel, labelpad=15.0, fontsize=28)
        ax.set_ylabel(ylabel, labelpad=15.0, fontsize=28)
        ax.set_title(title_list[i], fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # clb.ax.set_title(title_list[i], fontsize=24)
    # plt.savefig(f"figures/IDP/MaxEntIRL/2dplot_for_th1th2/e_agent_{name}.png")
    plt.show()


def learned_cost():
    env_type = "HPC"
    env_id = f"{env_type}_custom"
    name = "ext_sub01_deep_noreset_rewfirst"
    print(name)
    proj_path = os.path.abspath(os.path.join("..", "tmp", "log", env_id, "MaxEntIRL", name))
    with open(f"../demos/{env_type}/sub01.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    expt_trans = flatten_trajectories(expert_trajs)
    test_len = len(expert_trajs)
    pltqs = []
    if env_type == "HPC":
        for i in range(35):
            file = os.path.join("..", "demos", env_type, "sub01", f"sub01i{i + 1}.mat")
            pltqs += [io.loadmat(file)['pltq']]
        test_len = len(pltqs)
    venv = make_env(f"{env_id}-v1", use_vec_env=True, num_envs=1, pltqs=pltqs)
    th_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1))
    sample_until = make_sample_until(n_timesteps=None, n_episodes=test_len)
    i = 0
    cost_list = []
    while os.path.isdir(os.path.join(proj_path, "model", f"{i:03d}")):
        agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"))
        if os.path.isfile(os.path.abspath(proj_path + f"/{i:03d}/normalization.pkl")):
            stats_path = os.path.abspath(proj_path + f"/model/{i:03d}/normalization.pkl")
            venv = make_env("IDP_custom-v0", use_vec_env=True, num_envs=1, use_norm=True, stats_path=stats_path)
        agent_trajs = generate_trajectories(agent, venv, sample_until=sample_until, deterministic_policy=False)
        agent_trans = flatten_trajectories(agent_trajs)
        th_input = th.from_numpy(np.concatenate([agent_trans.obs, agent_trans.acts], axis=1)).double()
        with open(os.path.join(proj_path, "model", f"{i:03d}", "reward_net.pkl"), "rb") as f:
            reward_fn = pickle.load(f).double()
        print(f"{i:03d} Cost:", -reward_fn(th_input).sum().item() / test_len)
        cost_list.append(-reward_fn(th_input).sum().item() / test_len)
        i += 1
    plt.plot(cost_list)
    # plt.savefig(f"figures/IDP/MaxEntIRL/agent_cost_each_iter/expt_{name}.png")
    plt.show()


def expt_cost():
    def expt_fn(inp):
        return inp[:, :2].square().sum() + 0.1 * inp[:, 2:4].square().sum() + 1e-5 * (200 * inp[:, 6:]).square().sum()
    env_type = "HPC"
    env_id = f"{env_type}_pybullet"
    name = "cnn_sub01_deep_noreset_rewfirst"
    print(name)
    proj_path = os.path.abspath(os.path.join("..", "tmp", "log", env_id, "BC", name))
    with open(f"../demos/{env_type}/sub01.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    expt_trans = flatten_trajectories(expert_trajs)
    test_len = len(expert_trajs)
    pltqs = []
    if env_type == "HPC":
        for i in range(35):
            file = os.path.join("..", "demos", env_type, "sub01", f"sub01i{i+1}.mat")
            pltqs += [io.loadmat(file)['pltq']]
        test_len = len(pltqs)
    venv = make_env(f"{env_id}-v0", use_vec_env=True, num_envs=1, pltqs=pltqs)
    th_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1))
    print(f"expt_cost: {expt_fn(th_input).item() / test_len}")
    sample_until = make_sample_until(n_timesteps=None, n_episodes=test_len)
    i = 10
    cost_list = []
    while os.path.isdir(os.path.join(proj_path, "model", f"{i:03d}")):
        agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"), device='cpu')
        if os.path.isfile(proj_path + f"/{i:03d}/normalization.pkl"):
            stats_path = proj_path + f"/model/{i:03d}/normalization.pkl"
            venv = make_env(f"{env_id}-v0", use_vec_env=True, num_envs=1, use_norm=True, stats_path=stats_path, pltqs=pltqs)
        venv.reset()
        agent_trajs = generate_trajectories(agent, venv, sample_until=sample_until, deterministic_policy=True)
        agent_trans = flatten_trajectories(agent_trajs)
        th_input = th.from_numpy(np.concatenate([agent_trans.obs, agent_trans.acts], axis=1))
        print(f"{i:03d} Cost:", expt_fn(th_input).item() / test_len)
        cost_list.append(expt_fn(th_input).item() / test_len)
        i += 1
    # plt.plot(cost_list)
    # plt.savefig(f"figures/IDP/MaxEntIRL/expt_cost_each_iter/{name}.png")
    # plt.show()
    print(f"minimum cost index: {np.argmin(cost_list)}")


if __name__ == "__main__":
    def feature_fn(x):
        # return x
        # return x.square()
        return th.cat([x, x.square()], dim=1)
    expt_cost()
