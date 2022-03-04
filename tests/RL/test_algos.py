import os
import pytest
import pickle
import time
import numpy as np
from scipy import io

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from algos.tabular.viter import FiniteSoftQiter
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
    env_type = "DiscretizedHuman"
    name = f"{env_type}"
    subj = "sub06"
    irl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "IRL"))
    subpath = os.path.join(irl_dir, "demos", "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    with open(f"{irl_dir}/demos/DiscretizedHuman/17171719/{subj}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    init_states = []
    for traj in expt_trajs:
        init_states += [traj.obs[0]]
    env = make_env(f"{name}-v0", num_envs=1, N=[17, 17, 17, 19],
                   bsp=bsp, init_states=init_states, wrapper=DiscretizeWrapper)
    name += f"_{subj}_17171719"
    model_dir = os.path.join(rl_path, env_type, "tmp", "log", name, "softqiter", "policies_1")
    # model_dir = os.path.join("..", "..", "IRL", "tmp", "log")
    stats_path = None
    if os.path.isfile(model_dir + "normalization.pkl"):
        stats_path = model_dir + "normalization.pkl"
    with open(model_dir + "/agent.pkl", "rb") as f:
        algo = pickle.load(f)
    algo.set_env(env)
    for _ in range(5):
        obs = env.reset()
        done = False
        while not done:
            a, _ = algo.predict(obs, deterministic=False)
            ns, _, done, _ = env.step(a)
            env.render()
            time.sleep(env.get_attr("dt")[0])
            obs = ns
    env.close()
    # algo = SAC.load(model_dir + f"/agent")
    # a_list, o_list, _ = verify_policy(env, algo, render="human", repeat_num=10, deterministic=True)
    # plt.plot(a_list[0])
    # plt.show()
    # plt.plot(o_list[0][:, :2])
    # plt.show()


def test_finite_algo(rl_path):
    def feature_fn(x):
        return x ** 2
    env_type = "DiscretizedHuman"
    name = f"{env_type}"
    subj = "sub06"
    irl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "IRL"))
    subpath = os.path.join(irl_dir, "demos", "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    rwfn_dir = irl_dir + "/tmp/log/DiscretizedHuman/MaxEntIRL/sq_handnorm_19191919/sub06_1_1/model"
    with open(rwfn_dir + "/reward_net.pkl", "rb") as f:
        rwfn = CPU_Unpickler(f).load().to('cuda:0')
    rwfn.feature_fn = feature_fn
    env = make_env(f"{name}-v2", num_envs=1, N=[19, 19, 19, 19], NT=[11, 11], bsp=bsp,
                   wrapper=RewardInputNormalizeWrapper, wrapper_kwrags={'rwfn': rwfn})
    eval_env = make_env(f"{name}-v2", N=[19, 19, 19, 19], NT=[11, 11], bsp=bsp, wrapper=DiscretizeWrapper)
    agent = FiniteSoftQiter(env=env, gamma=1, alpha=0.001, device='cuda:0', verbose=True)
    agent.learn(0)
    agent.set_env(eval_env)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    for epi in range(10):
        ob = eval_env.reset()
        obs, acts, rews = agent.predict(ob, deterministic=False)
        ax1.plot(obs[:, :2])
        ax2.plot(obs[:, 2:])
        ax3.plot(acts)
        eval_env.render()
        for t in range(50):
            obs_idx = eval_env.get_idx_from_obs(ob)
            act_idx = agent.policy.choice_act(agent.policy.policy_table[t].T[obs_idx])
            act = eval_env.get_acts_from_idx(act_idx)
            ob, r, _, _ = eval_env.step(act[0])
            eval_env.render()
            time.sleep(eval_env.dt)
    plt.show()


def test_2d(rl_path):
    name = "2DTarget_disc"
    env_id = f"{name}"
    env = make_env(f"{env_id}-v0")
    model_dir = os.path.join(rl_path, name, "tmp", "log", env_id + "_more_random", "softqiter", "policies_1")
    with open(model_dir + "/agent.pkl", "rb") as f:
        algo = pickle.load(f)
    policy = []
    for i in range(5):
        policy.append(algo.policy.policy_table[:, 500 * i:500 * (i + 1)])
    plt.imshow(np.vstack(policy), cmap=cm.rainbow)
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
