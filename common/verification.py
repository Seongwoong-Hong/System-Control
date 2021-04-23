import os
import time
import cv2
import numpy as np
import torch as th
from typing import Dict, List, Union, Optional

from copy import deepcopy
from mujoco_py import GlfwContext
from matplotlib import pyplot as plt
from imitation.data.rollout import TrajectoryAccumulator, flatten_trajectories
from stable_baselines3.common.vec_env import DummyVecEnv


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
    def __init__(self, env, cost_fn, agent, expt: Optional = None):
        agent_env = DummyVecEnv([lambda: env])
        self.agent_dict = {'env': agent_env, 'cost_fn': cost_fn, 'algo': agent}
        self.agents = self.process_agent(self.agent_dict)
        if expt is not None:
            expt_env = DummyVecEnv([lambda: deepcopy(env)])
            self.expt_dict = {'env': expt_env, 'cost_fn': cost_fn, 'algo': expt}
            self.agents += self.process_agent(self.expt_dict)
        self.cost_accum = self.cal_cost(self.agents)
        self.fig = self.draw_costmap(self.cost_accum)

    @classmethod
    def process_agent(cls, agent_dict: Dict):
        trajectories = []
        env = agent_dict['env']
        algo = agent_dict['algo']
        trajectories_accum = TrajectoryAccumulator()
        obs = env.reset()
        for env_idx, ob in enumerate(obs):
            trajectories_accum.add_step(dict(obs=ob), env_idx)
        active = True
        while active:
            act, _ = algo.predict(obs, deterministic=False)
            obs, rew, done, info = env.step(act)
            done &= active
            new_trajs = trajectories_accum.add_steps_and_auto_finish(
                act, obs, rew, done, info
            )
            trajectories.extend(new_trajs)
            if env.env_method('exp_isend')[0]:
                active &= ~done
        orders = [i for i in range(len(trajectories))]
        cost_fn = agent_dict['cost_fn']
        transitions = [flatten_trajectories([traj]) for traj in trajectories]
        return [{'transitions': transitions, 'cost_fn': cost_fn, 'orders': orders}]

    @classmethod
    def cal_cost(cls, agents: List[Dict]):
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
                    done = tran.dones[t]
                    cost += cost_fn(obs, act, next_obs, done)
                costs.append(cost)
            orders = agent['orders']
            inputs.append([orders, costs])
        return inputs

    @classmethod
    def draw_costmap(cls, cost_accum: List[List]):
        fig = plt.figure()
        titles = ["agent", "expert"]
        for i, (orders, costs) in enumerate(cost_accum):
            ax = fig.add_subplot(len(cost_accum), 1, i + 1)
            ax.plot(orders, costs)
            ax.set_xlabel("order")
            ax.set_ylabel("cost")
            ax.set_title(titles[i])
        plt.show()
        return fig
