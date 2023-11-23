import os
import cv2
import numpy as np
import torch as th
from typing import Dict, List, Optional

from matplotlib import pyplot as plt
from imitation.data.rollout import TrajectoryAccumulator, flatten_trajectories, make_sample_until
from stable_baselines3.common.vec_env import DummyVecEnv


def video_record(imgs: List, filename: str, dt: float):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    frame_height = imgs[0].shape[0]
    frame_width = imgs[0].shape[1]
    fps = int(1 / dt)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
    for img in imgs:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        writer.write(img_bgr)
    writer.release()


def exec_policy(environment, policy, render="rgb_array", deterministic=True, repeat_num=5):
    if render == 'human':
        raise Exception("현재 환경에서 rendering은 지원하지 않습니다.")
    imgs = []
    acts_list = []
    obs_list = []
    rews_list = []
    for _ in range(repeat_num):
        actions = []
        rewards = []
        observs = []
        environment.render(mode=render)
        ob = environment.reset()
        observs.append(ob.squeeze())
        done = False
        img = environment.render(mode=render)
        imgs.append(img)
        while not done:
            act, _ = policy.predict(ob, deterministic=deterministic)
            ob, rew, done, info = environment.step(act)
            img = environment.render(mode=render)
            if hasattr(environment, "action") and callable(environment.action):
                act = environment.action(act)
            if type(info) == list:
                info = info[0]
            if "acts" in info:
                actions.append(info["acts"])
            else:
                actions.append(act.squeeze())
            observs.append(ob.squeeze())
            rewards.append(rew.squeeze())
            imgs.append(img)
        acts_list.append(np.array(actions))
        rews_list.append(np.array(rewards))
        obs_list.append(np.array(observs))
    return obs_list, acts_list, rews_list, imgs


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
        venv = agent_dict['env']
        n_episodes = 10
        if hasattr(venv, 'num_disturbs'):
            n_episodes = venv.num_disturbs
        sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes)
        if not isinstance(venv, DummyVecEnv):
            venv = DummyVecEnv([lambda: venv])
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
