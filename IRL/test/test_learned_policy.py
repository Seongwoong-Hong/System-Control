import gym_envs
import os
import time
import torch

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from IRL.project_policies import def_policy
from algo.torch.ppo import PPO
from common.wrappers import CostWrapper
from scipy import io


env_type = "HPC"
algo_type = "ppo"
name = "{}/{}/2021-3-15-20-29-54".format(env_type, algo_type)
num = 4
model_dir = os.path.join("..", "tmp", "log", name, "model")
# costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
# algo = PPO.load(model_dir + "/extra_ppo.zip")
algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
sub = "sub01"
expert_dir = os.path.join("..", "demos", env_type, sub + ".pkl")
pltqs = []
for i in range(35):
    file = os.path.join("..", "demos", env_type, sub, sub + "i%d.mat" % (i+1))
    pltqs += [io.loadmat(file)['pltq']]

env = gym_envs.make("{}_custom-v0".format(env_type), n_steps=300, pltqs=pltqs)
# env = CostWrapper(gym_envs.make("{}_custom-v0".format(env_type), n_steps=200), costfn)
exp = def_policy(env_type, env)
env.reset()
dt = env.dt
init_obs = env.reset().reshape(1, -1)
init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)
init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)
init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)

for iobs in init_obs:
    rew1_list = []
    rew2_list = []
    cost1_list = []
    cost2_list = []
    env.reset()
    env.set_state(iobs[:env.model.nq], iobs[env.model.nq:])
    obs = deepcopy(iobs)
    obs1_list = deepcopy(obs.reshape(1, -1))
    obs2_list = deepcopy(obs.reshape(1, -1))
    done = False
    # env.render()
    while not done:
        act, _ = exp.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        cost = obs @ exp.Q @ obs.T + (act * exp.gear) @ exp.R @ (act.T * exp.gear)
        # env.render()
        # rew1_list.append(act.item())
        cost1_list.append(cost.item())
        obs1_list = np.append(obs1_list, obs.reshape(1, -1), 0)
        time.sleep(dt)
    print(sum(cost1_list))

    env.reset()
    env.set_state(iobs[:env.model.nq], iobs[env.model.nq:])
    obs = deepcopy(iobs)
    done = False
    # env.render()
    while not done:
        act, _ = algo.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        cost = obs @ exp.Q @ obs.T + (act * exp.gear) @ exp.R @ (act.T * exp.gear)
        # env.render()
        # rew2_list.append(act.item())
        cost2_list.append(cost.item())
        obs2_list = np.append(obs2_list, obs.reshape(1, -1), 0)
        time.sleep(dt)
    print(sum(cost2_list))

    t = np.linspace(0, obs1_list.shape[0], obs1_list.shape[0])*0.02
    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    plt.ylim((-0.21, 0.06))
    plt.plot(t, obs1_list[:, 0])
    plt.plot(t, obs2_list[:, 0])
    plt.xlabel("time(s)")
    plt.ylabel(r"$\theta_1$(rad)")
    plt.axhline(y=0, color='k', linestyle="--")
    plt.ylim()
    plt.show()

    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    plt.ylim((-0.06, 0.21))
    plt.plot(t, obs1_list[:, 1])
    plt.plot(t, obs2_list[:, 1])
    plt.xlabel("time(s)")
    plt.ylabel(r"$\theta_2$(rad)")
    plt.axhline(y=0, color="k", linestyle="--")
    plt.show()
    # fig1 = plt.figure()
    # # ax1 = fig.add_subplot(1, 2, 1)
    # # ax2 = fig.add_subplot(1, 2, 2)
    # fig1.plot(obs1_list[:, 0])
    # fig1.plot(obs2_list[:, 0])
    #
    # fig2 = plt.figure()
    # fig2.plot(obs1_list[:, 1])
    # fig2.plot(obs2_list[:, 1])
    # plt.show()

env.close()
