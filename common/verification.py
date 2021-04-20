import os
import time
import cv2
import numpy as np

from copy import deepcopy
from mujoco_py import GlfwContext


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
