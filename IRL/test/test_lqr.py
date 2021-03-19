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
from common.wrappers import CostWrapper
from scipy import io


env_type = "HPC"
algo_type = "ppo"
name = "{}/{}/2021-3-15-20-29-54".format(env_type, algo_type)
num = 4
model_dir = os.path.join("..", "tmp", "log", name, "model")
sub = "sub01"
expert_dir = os.path.join("..", "demos", env_type, sub + ".pkl")
pltqs = []
for i in range(35):
    file = os.path.join("..", "demos", env_type, sub, sub + "i%d.mat" % (i+1))
    pltqs += [io.loadmat(file)['pltq']]

env = gym_envs.make("{}_custom-v0".format(env_type), n_steps=600, pltqs=pltqs)
ob_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
exp = def_policy(env_type, env, observation_space=ob_space)
env.reset()
dt = env.dt

for _ in range(5):
    rew1_list = []
    cost1_list = []
    obs = env.reset()
    obs1_list = deepcopy(obs[:4].reshape(1, -1))
    pT_list = deepcopy(obs[4:].reshape(1, -1))
    done = False
    # env.render()
    while not done:
        act, _ = exp.predict(obs[:4], deterministic=True)
        obs, rew, done, info = env.step(act)
        cost = obs[:4] @ exp.Q @ obs[:4].T + (act * exp.gear) @ exp.R @ (act.T * exp.gear)
        # env.render()
        # rew1_list.append(act.item())
        cost1_list.append(cost.item())
        obs1_list = np.append(obs1_list, obs[:4].reshape(1, -1), 0)
        pT_list = np.append(pT_list, obs[4:].reshape(1, -1), 0)
        time.sleep(dt)
    plt.plot(pT_list)
    plt.show()

env.close()
