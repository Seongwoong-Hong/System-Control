import os
import pytest
import pickle
import json
import time
import numpy as np
import torch as th
from scipy import io

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from algos.tabular.viter import FiniteSoftQiter, FiniteViter, SoftQiter
from common.util import make_env, CPU_Unpickler
from common.verification import verify_policy
from common.wrappers import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from stable_baselines3.common.vec_env import DummyVecEnv


@pytest.fixture
def rl_path():
    return os.path.abspath(os.path.join("..", "..", "RL"))


def test_mujoco_envs_learned_policy():
    env_name = "Pendulum"
    env = make_env(f"{env_name}-v0", use_vec_env=False)
    name = f"{env_name}/ppo"
    model_dir = os.path.join("..", "..", "RL", "mujoco_envs", "tmp", "log", name)
    algo = SAC.load(model_dir + "/agent")
    a_list, o_list, _ = verify_policy(env, algo, render='human', repeat_num=10)


def test_rl_learned_policy(rl_path):
    env_type = "Hopper"
    name = f"{env_type}"
    subj = "sub06"
    irl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "IRL"))
    subpath = os.path.join(irl_dir, "demos", "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    with open(f"{irl_dir}/demos/DiscretizedHuman/17171719/{subj}_1.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    init_states = []
    for traj in expt_trajs:
        init_states += [traj.obs[0]]
    # env = make_env(f"{name}-v0", bsp=bsp, init_states=init_states)#, wrapper=DiscretizeWrapper)
    env = make_env(f"{name}-v2", num_envs=1)
    # name += f"_{subj}"
    model_dir = os.path.join(rl_path, env_type, "tmp", "log", name, "sac", "policies_3")
    stats_path = None
    if os.path.isfile(model_dir + "normalization.pkl"):
        stats_path = model_dir + "normalization.pkl"
    # with open(model_dir + "/agent.pkl", "rb") as f:
    #     algo = pickle.load(f)
    # algo.set_env(env)
    algo = SAC.load(model_dir + f"/agent")
    lengths = []
    for _ in range(20):
        obs = env.reset()
        done = False
        length = 0
        while not done:
            a, _ = algo.predict(obs, deterministic=False)
            obs, _, done, _ = env.step(a)
            # env.render()
            # time.sleep(env.get_attr("dt")[0])
            length += 1
        print(length)
        lengths.append(length)
    print(np.mean(lengths), np.std(lengths))
    env.close()


def test_finite_algo(rl_path):
    name = "DiscretizedHuman"
    env_id = f"{name}"
    subj = "sub05"
    irl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "IRL"))
    subpath = os.path.join(irl_dir, "demos", "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    env = make_env(f"{env_id}-v2", N=[19, 19, 19, 19], NT=[11, 11], bsp=bsp, wrapper=DiscretizeWrapper)
    # init_states = np.array([[11, 35], [ 8, 16], [ 6,  2], [ 5, 45], [29, 27], [18, 37]])
    # env = make_env(f"{env_id}-v0", num_envs=1, init_states=init_states)
    # agent = SoftQiter(env=env, gamma=1, alpha=0.001, device='cuda:0')
    # agent.learn(50)
    agent = FiniteSoftQiter(env, gamma=1, alpha=0.01, device='cuda:0')
    agent.learn(0)
    for epi in range(300):
        # ob = init_states[epi % len(init_states)]
        ob = env.reset()
        obs, acts, rews = agent.predict(ob, deterministic=True)
        plt.plot(obs[:, 0], obs[:, 1])
        plt.xlim([0, 50])
        plt.ylim([0, 50])
        # plt.plot(obs[:, 1])
        # eval_env.render()
        # for t in range(50):
        #     obs_idx = eval_env.get_idx_from_obs(ob)
        #     act_idx = agent.policy.choice_act(agent.policy.policy_table[t].T[obs_idx])
        #     act = eval_env.get_acts_from_idx(act_idx)
        #     ob, r, _, _ = eval_env.step(act[0])
        #     eval_env.render()
        #     time.sleep(eval_env.dt)
    plt.show()


def test_toy_disc_env(rl_path):
    name = "SpringBall"
    env_id = f"{name}_disc"
    env = make_env(f"{env_id}-v2", wrapper=DiscretizeWrapper)
    algo = SoftQiter(env=env, gamma=0.99, alpha=0.01, device='cpu')
    algo.learn(1000)
    fig = plt.figure(figsize=[6.4, 6.4])
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    for _ in range(10):
        obs = []
        acts = []
        ob = env.reset()
        done = False
        # env.render()
        while not done:
            obs.append(ob)
            act, _ = algo.predict(ob, deterministic=False)
            ob, _, done, _ = env.step(act[0])
            acts.append(act[0])
            # env.render()
            # time.sleep(env.dt)
        obs = np.array(obs)
        acts = np.array(acts)
        ax1.plot(obs[:, 0])
        ax2.plot(obs[:, 1])
        ax3.plot(acts)
    fig.tight_layout()
    plt.show()


def test_1d(rl_path):
    name = "1DTarget_disc"
    env_id = f"{name}"
    map_size = 50
    env = make_env(f"{env_id}-v2", map_size=map_size, num_envs=1)
    model_dir = os.path.join(rl_path, name, "tmp", "log", env_id, "softqiter", "policies_1")
    with open(model_dir + "/agent.pkl", "rb") as f:
        algo = pickle.load(f)
    # algo = PPO.load(model_dir + "/agent")
    plt.imshow(algo.policy.policy_table, cmap=cm.rainbow)
    plt.show()
    print('end')


def test_total_reward(rl_path):
    env_type = "HPC"
    name = f"{env_type}_custom"
    model_dir = os.path.join(rl_path, env_type, "tmp", "log", name, "ppo", "policies_4")
    stats_path = None
    if os.path.isfile(model_dir + "normalization.pkl"):
        stats_path = model_dir + "normalization.pkl"
    env = make_env(f"{name}-v1", subpath="../../IRL/demos/HPC/sub01/sub01", wrapper=ActionWrapper, use_norm=stats_path)
    algo = PPO.load(model_dir + f"/agent")
    ob = env.reset()
    done = False
    actions = []
    obs = []
    rewards = []
    while not done:
        act, _ = algo.predict(ob, deterministic=False)
        ob, reward, done, info = env.step(act)
        rewards.append(reward)
        actions.append(info['acts'])
        obs.append(info['obs'])
    # plt.plot(np.array(actions).reshape(-1, 2))
    # plt.show()
    # plt.plot(np.array(obs).reshape(-1, 6)[:, :2])
    # plt.show()
    plt.plot(rewards)
    plt.show()
