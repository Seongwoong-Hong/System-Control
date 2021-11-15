import os
import pytest
import pickle
import numpy as np

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import verify_policy
from common.wrappers import ActionWrapper
import matplotlib.pyplot as plt


@pytest.fixture
def rl_path():
    return os.path.abspath(os.path.join("..", "..", "RL"))


def test_mujoco_envs_learned_policy():
    env_name = "Hopper"
    env = make_env(f"{env_name}-v2", use_vec_env=False)
    name = f"{env_name}/ppo"
    model_dir = os.path.join("..", "..", "RL", "mujoco_envs", "tmp", "log", name)
    algo = PPO.load(model_dir + "/policies_1/000002000000/model.pkl")
    a_list, o_list, _ = verify_policy(env, algo)


def test_rl_learned_policy(rl_path):
    env_type = "IDP"
    name = f"{env_type}_custom"
    model_dir = os.path.join(rl_path, env_type, "tmp", "log", name, "ppo", "policies_1")
    stats_path = None
    if os.path.isfile(model_dir + "normalization.pkl"):
        stats_path = model_dir + "normalization.pkl"
    env = make_env(f"{name}-v2", subpath="../../IRL/demos/HPC/sub01/sub01", wrapper=None, use_norm=stats_path)
    algo = PPO.load(model_dir + f"/agent")
    a_list, o_list, _ = verify_policy(env, algo, render="human", repeat_num=10, deterministic=True)
    plt.plot(a_list[0])
    plt.show()
    plt.plot(o_list[0][:, :2])
    plt.show()


def test_2d(rl_path):
    name = "2DTarget"
    env_id = f"{name}_disc"
    env = make_env(f"{env_id}-v2")
    model_dir = os.path.join(rl_path, name, "tmp", "log", env_id, "softqlearning", "policies_1")
    with open(model_dir + "/agent.pkl", "rb") as f:
        algo = pickle.load(f)
    # algo = PPO.load(model_dir + f"/agent")
    trajs = []
    for i in range(10):
        st = env.reset()
        done = False
        sts, rs = [], []
        while not done:
            action, _ = algo.predict(st, deterministic=False)
            st, r, done, _ = env.step(action)
            sts.append(st)
            rs.append(r)
        trajs.append(np.append(np.array(sts), np.array(rs).reshape(-1, 1), axis=1))
    env.draw(trajs)


def test_1d(rl_path):
    name = "1DTarget"
    env_id = f"{name}_disc"
    env = make_env(f"{env_id}-v0")
    model_dir = os.path.join(rl_path, name, "tmp", "log", env_id, "ppo", "policies_2")
    algo = PPO.load(model_dir + "/agent")
    a_list, o_list, _ = verify_policy(env, algo, render="None", repeat_num=12, deterministic=False)
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
