import os
import pickle

import torch as th
import numpy as np

from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import io
from imitation.algorithms import bc

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import CostMap, verify_policy
from common.wrappers import ActionWrapper


def draw_time_trajs(inp1, inp2, name=r"$\theta$s", labels=[4 + 5*i for i in range(7)]):
    t = np.linspace(0, 1/120 * (len(inp1[0])-1), len(inp1[0]))
    # ymax, ymin = 1, -1
    ymax, ymin = np.max(np.array(inp2)[:, :, :2]), np.min(np.array(inp2)[:, :, :2])
    plt.figure(figsize=[9, 6.4], dpi=600.0)
    for j in labels:
        yval_list = [inp1[j], inp2[j]]
        # plt.plot(yval_list[0][:, 0], yval_list[0][:, 1], color=(19 / 255, 0 / 255, 182 / 255, 1), lw=3)
        plt.plot(t, yval_list[0][:, 0], color=(19 / 255, 0 / 255, 182 / 255, 1), lw=3)
        plt.plot(t, yval_list[1][:, 0], color=(19 / 255, 0 / 255, 182 / 255, 0.4), lw=3)
        # plt.plot(yval_list[1][:, 0], yval_list[1][:, 1], color=(255 / 255, 105 / 255, 21 / 255, 1), lw=3)
        plt.plot(t, yval_list[0][:, 1], color=(255 / 255, 105 / 255, 21 / 255, 1), lw=3)
        plt.plot(t, yval_list[1][:, 1], color=(255 / 255, 105 / 255, 21 / 255, 0.6), lw=3)
        plt.legend(['', '', 'learned', 'original'], ncol=2, columnspacing=0.1, fontsize=15)
        # plt.legend(['learned', 'original'], ncol=2, columnspacing=0.1, fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.ylim(ymin, ymax)
        plt.xlim(np.min(t), np.max(t))
        # plt.xlim(-1, 1)
        plt.axhline(y=0.0, linestyle=':', color='0.5')
        plt.axvline(x=0.0, linestyle=':', color='0.5')
        plt.title("Simulation Result", fontsize=28, pad=30)
        plt.xlabel("time", fontsize=24)
        plt.ylabel(name, fontsize=24)
        # plt.savefig(f"figures/{env_type}/{subj}/angular_velocity{j}.png")
        plt.show()


def draw_trajectories():
    env_type = "HPC"
    algo_type = "MaxEntIRL"
    env_id = f"{env_type}_custom"
    subj = "sub01"
    wrapper = ActionWrapper if "HPC" in env_type else None
    # pltqs, init_states = [], []
    # for i in range(5, 10):
    #     pltqs += [io.loadmat(f"../demos/HPC/sub01/sub01i{i+1}.mat")['pltq']]
    #     init_states += [io.loadmat(f"../demos/HPC/sub01/sub01i{i+1}.mat")['state'][0, :4]]
    env = make_env(f"{env_id}-v0", wrapper=wrapper, use_vec_env=False, subpath=f"../demos/HPC/sub01/sub01")
    # env = make_env(f"{env_type}-v0", wrapper=wrapper, pltqs=pltqs, init_states=init_states)
    name = f"{env_id}/{algo_type}/ext_{subj}_linear_ppoagent_svm_reset"
    model_dir = os.path.join("..", "tmp", "log", name, "model", "002")
    with open(f"../demos/{env_type}/{subj}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    lnum = len(expert_trajs)
    expt_obs = [expert_trajs[i].obs for i in range(lnum)]
    expt_acts = [expert_trajs[i].acts for i in range(lnum)]
    # algo = SAC.load("../../RL/2DWorld/tmp/log/2DWorld/sac/policies_4/agent.zip")
    # algo = bc.reconstruct_policy("../../tests/algos/policy")
    algo = PPO.load(model_dir + "/agent")
    agent_acts, agent_obs, _ = verify_policy(env, algo, deterministic=False, render="None", repeat_num=lnum)
    draw_time_trajs(agent_obs, expt_obs)
    # draw_time_trajs(agent_acts, expt_acts, name="actions", labels=[i for i in range(35)])


def draw_costfigure():
    # def expt_reward(inp):
    #     x, y = inp[:, 0], inp[:, 1]
    #     return th.exp(-0.5 * (x ** 2 + y ** 2)) \
    #         - th.exp(-0.5 * ((x - 5 / 2) ** 2 + (y - 5 / 2) ** 2)) \
    #         - th.exp(-0.5 * ((x + 5 / 2) ** 2 + (y - 5 / 2) ** 2)) \
    #         - th.exp(-0.5 * ((x - 5 / 2) ** 2 + (y + 5 / 2) ** 2)) \
    #         - th.exp(-0.5 * ((x + 5 / 2) ** 2 + (y + 5 / 2) ** 2))
    def expt_reward(inp):
        x, y = inp[:, 0], inp[:, 1]
        return -((x - 2/3) ** 2 + (y - 2/3) ** 2)
    env_type = "2DTarget"
    env_id = f"{env_type}"
    subj = "ppo"
    subpath = os.path.abspath(os.path.join("..", "demos", env_type, subj))
    env = make_env(f"{env_id}-v0", use_vec_env=False, subpath=subpath + f"/{subj}")
    algo_type = "MaxEntIRL"
    name = f"ext_{subj}_linear_ppoagent_svm_reset_0.2"
    num = 8
    load_dir = os.path.abspath(f"../tmp/log/{env_id}/{algo_type}/{name}/model")
    algo = PPO.load(load_dir + f"/{num:03d}/agent")
    # algo = def_policy("IDP", env)
    # algo = PPO.load(f"../../RL/IDP/tmp/log/IDP_custom/ppo/policies_1/ppo0")
    with open(load_dir + f"/{num:03d}/reward_net.pkl", "rb") as f:
        reward_fn = pickle.load(f).double()

    ndim = env.observation_space.shape[0]
    d1, d2 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    pact = np.zeros((100, 100), dtype=np.float64)
    cost1, cost2 = np.zeros(d1.shape), np.zeros(d1.shape)
    for i in range(d1.shape[0]):
        for j in range(d1.shape[1]):
            iobs = np.zeros(ndim)
            iobs[0], iobs[1] = deepcopy(d1[i][j]), deepcopy(d2[i][j])
            iacts, _ = algo.predict(np.array(iobs), deterministic=True)
            # pact[i][j] = iacts[0]
            # inp = th.from_numpy(np.append(iobs, iacts)).double().to(algo.device).reshape(1, -1)
            inp = th.from_numpy(iobs).double().to(algo.device).reshape(1, -1)
            cost1[i][j] = -expt_reward(inp)
            cost2[i][j] = -reward_fn(inp).item()
    cost1 = (cost1 - np.min(cost1)) / (np.max(cost1) - np.min(cost1))
    cost2 = (cost2 - np.min(cost2)) / (np.max(cost2) - np.min(cost2))
    title_list = ["original cost", "learned cost"]
    yval_list = [cost1, cost2]
    xlabel, ylabel = r"$\theta_1$", r"$\theta_2$"
    max_list = [1.0, 1.0]
    min_list = [0.0, 0.0]
    fig = plt.figure(figsize=[12, 5.8], dpi=300.0)
    for i in [0, 1]:
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        # d1, d2 = np.meshgrid(d1, d2)
        surf = ax.plot_surface(d1, d2, yval_list[i], rstride=1, cstride=1, cmap=cm.rainbow,
                               vmax=max_list[i], vmin=min_list[i])
        # clb = fig.colorbar(surf, ax=ax)
        ax.set_xlabel(xlabel, labelpad=15.0, fontsize=28)
        ax.set_ylabel(ylabel, labelpad=15.0, fontsize=28)
        ax.set_title(title_list[i], fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.view_init(elev=90, azim=0)
        # clb.ax.set_title(title_list[i], fontsize=24)
    # plt.savefig("check.png")
    plt.show()


def draw_costmaps():
    env_type = "IDP"
    algo_type = "MaxEntIRL"
    device = "cpu"
    name = "IDP_custom"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = make_env(f"{name}-v0", use_vec_env=False, n_steps=600, subpath="sub01")
    expt_env = make_env(f"{name}-v2", use_vec_env=False, n_steps=600, subpath="sub01")
    ana_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, name + "_lqr")
    model_dir = os.path.join(ana_dir, "model")
    with open(model_dir + "/reward_net.pkl", "rb") as f:
        reward_net = pickle.load(f).double()

    def cost_fn(*args):
        inp = th.cat([args[0], args[1]], dim=1)
        return -reward_net(inp).item()
    agent = SAC.load(model_dir + "/agent.zip")
    # expt = def_policy(env_type, expt_env)
    expt = PPO.load(f"../../RL/{env_type}/tmp/log/{name}/ppo/policies_1/ppo0")
    # expt = PPO.load(f"tmp/log/{env_type}/ppo/forward/model/extra_ppo0.zip")
    cost_map = CostMap(cost_fn, env, agent, expt_env, expt)


if __name__ == "__main__":
    def feature_fn(x):
        # return x
        return th.cat([x, x**2, x**3, x**4], dim=1)
    # draw_costfigure()
    draw_trajectories()
