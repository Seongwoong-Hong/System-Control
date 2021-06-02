import os
import time
import cv2
import numpy as np
import torch as th
from typing import Dict, List, Optional

from copy import deepcopy
from mujoco_py import GlfwContext
from matplotlib import pyplot as plt
from imitation.data.rollout import TrajectoryAccumulator, flatten_trajectories, make_sample_until
from stable_baselines3.common.vec_env import DummyVecEnv


def video_record(imgs: List, filename: str, dt: float):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    width, height, _ = imgs[0].shape
    writer = cv2.VideoWriter(filename, fourcc, 1 / dt, (width, height))
    for img1 in imgs:
        img = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        writer.write(img)


def verify_policy(environment, policy, render="human", deterministic=True, repeat_num=5):
    if render == 'rgb_array':
        GlfwContext(offscreen=True)
    imgs = []
    acts_list = []
    obs_list = []
    for _ in range(repeat_num):
        actions = np.zeros((1,) + environment.action_space.shape)
        ob = environment.reset()
        obs = deepcopy(ob.reshape(1, -1))
        done = False
        img = environment.render(mode=render)
        imgs.append(img)
        while not done:
            act, _ = policy.predict(ob, deterministic=deterministic)
            ob, rew, done, info = environment.step(act)
            img = environment.render(mode=render)
            actions = np.append(actions, act.reshape(1, -1), 0)
            obs = np.append(obs, ob.reshape(1, -1), 0)
            time.sleep(environment.dt)
            imgs.append(img)
        acts_list.append(actions)
        obs_list.append(obs)
    return acts_list, obs_list, imgs


class CostMap:
    def __init__(self, cost_fn, agent_env, agent, expt_env: Optional = None, expt: Optional = None):
        self.agent_dict = {'env': agent_env, 'cost_fn': cost_fn, 'algo': agent}
        self.agents = self.process_agent(self.agent_dict)
        if expt is not None:
            self.expt_dict = {'env': expt_env, 'cost_fn': cost_fn, 'algo': expt}
            self.agents += self.process_agent(self.expt_dict)
        self.cost_accum = self.cal_cost(self.agents)
        self.fig = self.draw_costmap(self.cost_accum)

    @classmethod
    def process_agent(cls, agent_dict: Dict):
        trajectories = []
        env = agent_dict['env']
        n_episodes = 10
        if hasattr(env, 'num_disturbs'):
            n_episodes = env.num_disturbs
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes)
        venv = DummyVecEnv([lambda: env])
        algo = agent_dict['algo']
        trajectories_accum = TrajectoryAccumulator()
        obs = venv.reset()
        for env_idx, ob in enumerate(obs):
            trajectories_accum.add_step(dict(obs=ob), env_idx)
        active = True
        while active:
            act, _ = algo.predict(obs, deterministic=False)
            obs, rew, done, info = venv.step(act)
            done &= active
            new_trajs = trajectories_accum.add_steps_and_auto_finish(
                act, obs, rew, done, info
            )
            trajectories.extend(new_trajs)
            if sample_until(trajectories):
                active &= ~done
        orders = [i+1 for i in range(len(trajectories))]
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
                obs = th.from_numpy(tran.obs)
                act = th.from_numpy(tran.acts)
                next_obs = th.from_numpy(tran.next_obs)
                done = tran.dones
                cost = cost_fn(obs, act, next_obs, done).sum().item()
                costs.append(cost)
            orders = agent['orders']
            inputs.append([orders, costs])
        return inputs

    @classmethod
    def draw_costmap(cls, cost_accum: List[List]):
        from matplotlib.ticker import FormatStrFormatter
        fig = plt.figure(figsize=(7.5, 5.4*len(cost_accum)))
        titles = ["agent", "expert"]
        for i, (orders, costs) in enumerate(cost_accum):
            ax = fig.add_subplot(len(cost_accum), 1, i + 1)
            ax.plot(orders, costs)
            ax.grid()
            ax.set_xlabel("#", fontsize=20)
            ax.set_ylabel("cost", fontsize=20)
            ax.set_title(titles[i], fontsize=23)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
        plt.show()
        return fig
