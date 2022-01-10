import os
import pytest
import pickle
import numpy as np
from scipy import io

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import verify_policy
from common.wrappers import ActionWrapper
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
    env_type = "DiscretizedDoublePendulum"
    name = f"{env_type}"
    subj = "sub07"
    model_dir = os.path.join(rl_path, env_type, "tmp", "log", name + f"_{subj}_init", "softqiter", "policies_1")
    # model_dir = os.path.join("..", "..", "IRL", "tmp", "log")
    stats_path = None
    if os.path.isfile(model_dir + "normalization.pkl"):
        stats_path = model_dir + "normalization.pkl"
    wrapper = ActionWrapper if env_type == "HPC" else None
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "IRL"))
    subpath = os.path.join(proj_path, "demos", "HPC", f"{subj}_cropped", subj)
    init_states = []
    for i in range(5):
        for j in range(6):
            bsp = io.loadmat(subpath + f"i{i + 1}_{j}.mat")['bsp']
            init_states += [io.loadmat(subpath + f"i{i + 1}_{j}.mat")['state'][0, :4]]
    env = make_env(f"{name}-v2", num_envs=1, h=[0.03, 0.03, 0.05, 0.08])
    # env = make_env(f"{name}-v2", subpath="../../IRL/demos/HPC/sub01/sub01", wrapper=wrapper, use_norm=stats_path)
    with open(model_dir + "/agent.pkl", "rb") as f:
        algo = pickle.load(f)
    algo.set_env(env)
    import time
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
