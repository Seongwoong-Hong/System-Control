import gym_envs
import gym
import os
import time
import torch

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from IRL.project_policies import def_policy
from algo.torch.ppo import PPO
from algo.torch.sac import SAC
from common.wrappers import CostWrapper
from scipy import io


def according_policy(environment, policy):
    action_list = np.zeros([1, 2])
    obs = environment.reset()
    ob_list = deepcopy(obs.reshape(1, -1))
    done = False
    environment.render()
    while not done:
        act, _ = policy.predict(obs, deterministic=True)
        obs, rew, done, info = environment.step(act)
        environment.render()
        action_list = np.append(action_list, act.reshape(1, -1), 0)
        ob_list = np.append(ob_list, obs.reshape(1, -1), 0)
        time.sleep(environment.dt)
    return action_list, ob_list


if __name__=="__main__":
    env_type = "HPC"
    algo_type = "sac"
    name = "{}/{}/2021-3-23-13-52-55".format(env_type, algo_type)
    num = 14
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    # costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
    # algo = PPO.load(model_dir + "/extra_ppo.zip")
    # algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
    algo = SAC.load(model_dir + "/ppo{}.zip".format(num))
    sub = "sub01"
    expert_dir = os.path.join("..", "demos", env_type, sub + ".pkl")
    pltqs = []
    for i in range(35):
        file = os.path.join("..", "demos", env_type, sub, sub + "i%d.mat" % (i+1))
        pltqs += [io.loadmat(file)['pltq']]

    env = gym_envs.make("{}_custom-v0".format(env_type), n_steps=600, pltqs=pltqs, order=[i for i in range(35)])
    # env = CostWrapper(gym_envs.make("{}_custom-v0".format(env_type), n_steps=200), costfn)
    ob_space = gym.spaces.Box(-10, 10, (4, ))
    exp = def_policy(env_type, env, observation_space=ob_space)
    init_obs = env.reset().reshape(1, -1)
    init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)
    init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)
    init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)

    for i in range(35):
        # act1_list, st1_list = according_policy(env, exp)
        act_list, st_list = according_policy(env, algo)
        pltq_list = st_list[:, 4:]
        obs_list = st_list[:, :4]

        # t = np.linspace(0, obs2_list.shape[0], obs2_list.shape[0])*0.02
        # plt.rc("axes", labelsize=18)
        # plt.rc("xtick", labelsize=15)
        # plt.rc("ytick", labelsize=15)
        # plt.ylim((-0.21, 0.06))
        # # plt.plot(t, obs1_list[:, 0])
        # plt.plot(t, obs2_list[:, 0])
        # plt.xlabel("time(s)")
        # plt.ylabel(r"$\theta_1$(rad)")
        # plt.axhline(y=0, color='k', linestyle="--")
        # plt.ylim()
        # plt.show()
        #
        # plt.rc("axes", labelsize=18)
        # plt.rc("xtick", labelsize=15)
        # plt.rc("ytick", labelsize=15)
        # plt.ylim((-0.06, 0.21))
        # # plt.plot(t, obs1_list[:, 1])
        # plt.plot(t, obs2_list[:, 1])
        # plt.xlabel("time(s)")
        # plt.ylabel(r"$\theta_2$(rad)")
        # plt.axhline(y=0, color="k", linestyle="--")
        # plt.show()
        # fig1 = plt.figure()
        # fig2 = plt.figure()
        # ax1 = fig.add_subplot(1, 2, 1)
        # ax2 = fig.add_subplot(1, 2, 2)
        # fig1.plot(obs1_list)
        plt.plot(obs_list[:, :2])
        plt.show()
        plt.plot(act_list)
        plt.show()
        # plt.plot(pltq_list)
        # fig2 = plt.figure()
        # fig2.plot(obs1_list[:, 1])
        # fig2.plot(obs2_list[:, 1])
        # plt.show()

    env.close()
