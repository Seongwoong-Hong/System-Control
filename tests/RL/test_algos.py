import os
import pytest
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
    env_type = "HPC"
    name = f"{env_type}_custom"
    model_dir = os.path.join(rl_path, env_type, "tmp", "log", name, "ppo", "policies_4")
    stats_path = None
    if os.path.isfile(model_dir + "normalization.pkl"):
        stats_path = model_dir + "normalization.pkl"
    env = make_env(f"{name}-v1", subpath="../../IRL/demos/HPC/sub01/sub01", wrapper=ActionWrapper, use_norm=stats_path)
    algo = PPO.load(model_dir + f"/agent")
    a_list, o_list, _ = verify_policy(env, algo, render="human", repeat_num=10, deterministic=True)
    plt.plot(a_list[0])
    plt.show()
    plt.plot(o_list[0][:, :2])
    plt.show()


def test_total_reward(rl_path):
    env_type = "HPC"
    name = f"{env_type}_custom"
    model_dir = os.path.join(rl_path, env_type, "tmp", "log", name, "ppo", "policies_4")
    stats_path = None
    if os.path.isfile(model_dir + "normalization.pkl"):
        stats_path = model_dir + "normalization.pkl"
    env = make_env(f"{name}-v0", subpath="../../IRL/demos/HPC/sub01/sub01", wrapper=ActionWrapper, use_norm=stats_path)
    algo = PPO.load(model_dir + f"/agent")
    ob = env.reset()
    done = False
    actions = []
    obs = []
    while not done:
        act, _ = algo.predict(ob, deterministic=False)
        ob, reward, done, info = env.step(act)
        actions.append(info['acts'])
        obs.append(info['obs'])
    plt.plot(np.array(actions).reshape(-1, 2))
    plt.show()
    plt.plot(np.array(obs).reshape(-1, 6)[:, :2])
    plt.show()
