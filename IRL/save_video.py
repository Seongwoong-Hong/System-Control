import cv2
import gym
import gym_envs
import os
import time
import torch

import numpy as np
from mujoco_py import GlfwContext
from copy import deepcopy

from IRL.project_policies import def_policy
from algo.torch.ppo import PPO
from common.wrappers import CostWrapper


def video_record(imgs, filename, dt):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    width, height, _ = imgs[0].shape
    writer = cv2.VideoWriter(filename, fourcc, 1 / dt, (width, height))
    for img1 in imgs:
        img = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        writer.write(img)


GlfwContext(offscreen=True)
env_type = "IDP"
name = "{}/2021-1-29-12-52-56".format(env_type)
num = 14
model_dir = os.path.join("tmp", "log", name, "model")
costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
# algo = PPO.load(model_dir + "/extra_ppo.zip")
env = CostWrapper(gym.make("{}_custom-v1".format(env_type), n_steps=200), costfn)
exp = def_policy(env_type, env)
dt = env.dt
init_obs = env.reset().reshape(1, -1)
init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)
init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)
init_obs = np.append(init_obs, env.reset().reshape(1, -1), 0)
imgs1, imgs2 = [], []

for iobs in init_obs:
    env.reset()
    env.set_state(iobs[:env.model.nq], iobs[env.model.nq:])
    imgs1.append(env.render("rgb_array"))
    done = False
    obs = deepcopy(iobs)
    while not done:
        act, _ = exp.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        imgs1.append(env.render("rgb_array"))
        time.sleep(dt)
    del obs

    env.reset()
    env.set_state(iobs[:env.model.nq], iobs[env.model.nq:])
    imgs2.append(env.render("rgb_array"))
    done = False
    obs = deepcopy(iobs)
    while not done:
        act, _ = algo.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        cost = obs @ exp.Q @ obs.T + (act * exp.gear) @ exp.R @ (act.T * exp.gear)
        imgs2.append(env.render("rgb_array"))
        time.sleep(dt)

env.close()

video_record(imgs1, "videos/{}_expert.avi".format(name), dt)
video_record(imgs2, "videos/{}_agent.avi".format(name), dt)
