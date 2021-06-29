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
from IRL.scripts.project_policies import def_policy


def draw_2dfigure():
    env = make_env("IDP_custom-v0", use_vec_env=False)
    name = "extcnn_lqr_ppo_ppoagent_deep_noreset"
    num = "019"
    algo = PPO.load(f"../tmp/log/IDP_pybullet/MaxEntIRL/{name}/model/{num}/agent")
    # algo = def_policy("IDP", env)
    # algo = PPO.load(f"../../RL/IDP/tmp/log/IDP_custom/ppo/policies_1/ppo0")
    with open(f"../tmp/log/IDP_pybullet/MaxEntIRL/{name}/model/{num}/reward_net.pkl", "rb") as f:
        reward_fn = pickle.load(f).double()

    ndim, nact = env.observation_space.shape[0], env.action_space.shape[0]
    d1, d2 = np.meshgrid(np.linspace(-0.5, 0.5, 100), np.linspace(-0.25, 0.25, 100))
    pact = np.zeros((100, 100), dtype=np.float64)
    cost = np.zeros(d1.shape)
    for i in range(d1.shape[0]):
        for j in range(d1.shape[1]):
            iobs = np.zeros(ndim)
            iobs[0], iobs[2] = deepcopy(d1[i][j]), deepcopy(d2[i][j])
            iacts, _ = algo.predict(np.array(iobs), deterministic=True)
            pact[i][j] = iacts[1]
            inp = th.from_numpy(np.append(iobs, iacts)).double().to(algo.device).reshape(1, -1)
            cost[i][j] = th.sum(th.square(inp[0, :2]) + 0.1*th.square(inp[0, 2:4])+1e-5*th.square(200*inp[0, 4:])).item()
            # cost[i][j] = -reward_fn(inp).item()
    cost = (cost - np.min(cost))/(np.max(cost) - np.min(cost))
    title_list = ["norm_cost", "abs_action"]
    yval_list = [cost, np.abs(pact)]
    xlabel, ylabel = "d1", "d2"
    max_list = [1.0, 0.5]
    min_list = [0.0, 0.0]
    fig = plt.figure()
    for i in range(2):
        ax = fig.add_subplot(1, 2, (i + 1))
        surf = ax.pcolor(d1, d2, yval_list[i], cmap=cm.coolwarm, shading='auto', vmax=max_list[i],
                         vmin=min_list[i])
        clb = fig.colorbar(surf, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        clb.ax.set_title(title_list[i])
    # plt.savefig(f"figures/IDP/MaxEntIRL/2dplot_for_th1th2/e_agent_{name}.png")
    plt.show()


def learned_cost():
    name = "no_lqr_ppo_ppoagent_noreset"
    proj_path = os.path.abspath(os.path.join("..", "tmp", "log", "IDP_custom", "MaxEntIRL", name))
    with open("../demos/IDP/lqr_ppo.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    expt_trans = flatten_trajectories(expert_trajs)
    venv = make_env("IDP_custom-v0", use_vec_env=True, num_envs=1)
    th_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1))
    sample_until = make_sample_until(n_timesteps=None, n_episodes=10)
    i = 1
    cost_list = []
    while os.path.isdir(os.path.join(proj_path, "model", f"{i:03d}")):
        agent = PPO.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"))
        if os.path.isfile(os.path.abspath(proj_path + f"/{i:03d}/normalization.pkl")):
            stats_path = os.path.abspath(proj_path + f"/model/{i:03d}/normalization.pkl")
            venv = make_env("IDP_custom-v0", use_vec_env=True, num_envs=1, use_norm=True, stats_path=stats_path)
        agent_trajs = generate_trajectories(agent, venv, sample_until=sample_until, deterministic_policy=False)
        agent_trans = flatten_trajectories(agent_trajs)
        th_input = th.from_numpy(np.concatenate([agent_trans.obs, agent_trans.acts], axis=1)).double()
        with open(os.path.join(proj_path, "model", f"{i:03d}", "reward_net.pkl"), "rb") as f:
            reward_fn = pickle.load(f).double()
        print("Cost:", -reward_fn(th_input).mean().item() * 600)
        cost_list.append(-reward_fn(th_input).mean().item() * 600)
        i += 1
    plt.plot(cost_list)
    # plt.savefig(f"figures/IDP/MaxEntIRL/agent_cost_each_iter/expt_{name}.png")
    plt.show()


def expt_cost():
    def expt_fn(inp):
        return inp[:, :2].square() + 1e-5 * inp[:, 4:].square()
    env_type = "IDP"
    env_id = "IDP_pybullet"
    name = "cnn_lqr_ppo_noreset"
    proj_path = os.path.abspath(os.path.join("..", "tmp", "log", env_id, "MaxEntIRL", name))
    with open(f"../demos/{env_type}/lqr_ppo.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    expt_trans = flatten_trajectories(expert_trajs)
    test_len = 10
    pltqs = []
    if env_type == "HPC":
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            file = os.path.join("..", "demos", env_type, "sub01", f"sub01i{i+1}.mat")
            pltqs += [io.loadmat(file)['pltq']]
        test_len = len(pltqs)
    venv = make_env(f"{env_type}_custom-v0", use_vec_env=True, num_envs=1, pltqs=pltqs)
    th_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1))
    print(f"expt_cost: {expt_fn(th_input).mean().item() * 600}")
    sample_until = make_sample_until(n_timesteps=None, n_episodes=test_len)
    i = 0
    cost_list = []
    while os.path.isdir(os.path.join(proj_path, "model", f"{i:03d}")):
        agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"), device='cpu')
        if os.path.isfile(proj_path + f"/{i:03d}/normalization.pkl"):
            stats_path = proj_path + f"/model/{i:03d}/normalization.pkl"
            venv = make_env(f"{env_type}_pybullet-v0", use_vec_env=True, num_envs=1, use_norm=True, stats_path=stats_path, pltqs=pltqs)
        venv.reset()
        agent_trajs = generate_trajectories(agent, venv, sample_until=sample_until, deterministic_policy=False)
        agent_trans = flatten_trajectories(agent_trajs)
        th_input = th.from_numpy(np.concatenate([agent_trans.obs, agent_trans.acts], axis=1))
        print("Cost:", expt_fn(th_input).mean().item() * 600)
        cost_list.append(expt_fn(th_input).mean().item() * 600)
        i += 1
    plt.plot(cost_list)
    # plt.savefig(f"figures/IDP/MaxEntIRL/expt_cost_each_iter/{name}.png")
    plt.show()
    print(f"minimum cost index: {np.argmin(cost_list) + 1}")


if __name__ == "__main__":
    def feature_fn(x):
        return x
    expt_cost()
