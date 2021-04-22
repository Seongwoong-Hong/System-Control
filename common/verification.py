import os
import time
import cv2
import numpy as np
import torch as th

from copy import deepcopy
from mujoco_py import GlfwContext
from matplotlib import pyplot as plt


def video_record(imgs, filename, dt):
    paths = []
    parent_path = os.path.abspath(os.path.join(filename, os.pardir))
    while not os.path.isdir(parent_path):
        paths.append(parent_path)
        parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
    for path in reversed(paths):
        os.mkdir(path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    width, height, _ = imgs[0].shape
    writer = cv2.VideoWriter(filename, fourcc, 1 / dt, (width, height))
    for img1 in imgs:
        img = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        writer.write(img)


def verify_policy(environment, policy, render="human"):
    if render == 'rgb_array':
        GlfwContext(offscreen=True)
    imgs = []
    action_list = np.zeros((1,) + environment.action_space.shape)
    obs = environment.reset()
    ob_list = deepcopy(obs.reshape(1, -1))
    done = False
    img = environment.render(mode=render)
    imgs.append(img)
    while not done:
        act, _ = policy.predict(obs, deterministic=True)
        obs, rew, done, info = environment.step(act)
        img = environment.render(mode=render)
        action_list = np.append(action_list, act.reshape(1, -1), 0)
        ob_list = np.append(ob_list, obs.reshape(1, -1), 0)
        time.sleep(environment.dt)
        imgs.append(img)
    return action_list, ob_list, imgs


class CostMap:
    def __init__(self, agent, expt=None):
        self.agent = agent
        self.expt = expt

    @staticmethod
    def cal_cost(agents):
        inputs = []
        for agent in agents:
            transitions = agent['transitions']
            cost_fn = agent['cost_fn']
            costs = []
            for tran in transitions:
                cost = 0
                for t in range(len(tran)):
                    obs = th.from_numpy(tran.obs[np.newaxis, t, :])
                    act = th.from_numpy(tran.acts[np.newaxis, t, :])
                    next_obs = th.from_numpy(tran.next_obs[np.newaxis, t, :])
                    cost += cost_fn(obs, act, next_obs, tran.dones[t])
                costs.append(cost)
            orders = agent['orders']
            inputs.append([orders, costs])
        return inputs

    @staticmethod
    def draw_costmap(inputs):
        fig = plt.figure()
        titles = ["agent", "expert"]
        for i, (orders, costs) in enumerate(inputs):
            ax = fig.add_subplot(len(inputs), 1, i+1)
            ax.plot(orders, costs)
            ax.set_xlabel("order")
            ax.set_ylabel("cost")
            ax.set_title(titles[i])
        plt.show()
        return fig
