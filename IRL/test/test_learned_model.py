import gym
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


env_type = "IDP"
name = "{}/2021-1-29-12-52-56".format(env_type)
num = 14
model_dir = os.path.join("..", "tmp", "log", name, "model")
costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
# algo = PPO.load(model_dir + "/extra_ppo.zip")
algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
env = CostWrapper(gym.make("{}_custom-v1".format(env_type), n_steps=200), costfn)
exp = def_policy(env_type, env)
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
    env.render()
    while not done:
        act, _ = exp.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        cost = obs @ exp.Q @ obs.T + (act * exp.gear) @ exp.R @ (act.T * exp.gear)
        env.render()
        # rew1_list.append(act.item())
        cost1_list.append(cost.item())
        obs1_list = np.append(obs1_list, obs.reshape(1, -1), 0)
        time.sleep(dt)
    print(obs)

    env.reset()
    env.set_state(iobs[:env.model.nq], iobs[env.model.nq:])
    obs = deepcopy(iobs)
    done = False
    env.render()
    while not done:
        act, _ = algo.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        cost = obs @ exp.Q @ obs.T + (act * exp.gear) @ exp.R @ (act.T * exp.gear)
        # rew2_list.append(act.item())
        cost2_list.append(cost.item())
        obs2_list = np.append(obs2_list, obs.reshape(1, -1), 0)
        time.sleep(dt)
    print(obs)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(obs1_list[:, 0])
    ax2.plot(obs2_list[:, 0])
    ax1.plot(obs1_list[:, 1])
    ax2.plot(obs1_list[:, 1])
    plt.show()

env.close()
