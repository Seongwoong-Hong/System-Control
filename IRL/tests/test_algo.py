import os
import gym_envs

from algo.torch.ppo import PPO
from IRL.project_policies import def_policy
from common.verification import verify_policy
from matplotlib import pyplot as plt


def test_hpc_algo(env):
    algo = def_policy("HPC", env)
    a_list, o_list, _ = verify_policy(env, algo)


def test_hpcdiv_algo(tenv):
    algo = def_policy("HPCDiv", tenv)
    for _ in range(35):
        a_list, o_list, _ = verify_policy(tenv, algo)


def test_hpc_learned_policy(env):
    name = "HPC/ppo/AIRL_test/" + "81"
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    algo = PPO.load(model_dir + "/gen.zip")
    for _ in range(10):
        a_list, o_list, _ = verify_policy(env, algo)


def test_idp_learned_policy():
    env = gym_envs.make("IDP_custom-v2", n_steps=600)
    name = "IDP/ppo/AIRL_easy/" + "16"
    # name = "IDP/ppo/forward"
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    algo = PPO.load(model_dir + "/gen.zip")
    a_list, o_list, _ = verify_policy(env, algo)


def test_idp_policy():
    env = gym_envs.make("IDP_custom-v0", n_steps=600)
    algo = def_policy("IDP", env)
    _, _, _ = verify_policy(env, algo)


def test_mujoco_policy():
    import gym
    name = "InvertedDoublePendulum"
    env = gym.make(f"{name}-v2")
    # model_dir = os.path.join("..", "tmp", "log", "mujoco_envs", "ppo", name, "sample", "checkpoints", "final", "gen_policy")
    model_dir = os.path.join("..", "tmp", "log", "mujoco_envs", "ppo", name+"2", "17", "model")
    # model_dir = os.path.join("..", "..", "RL", "mujoco_envs", "tmp", "log", name, "ppo")
    algo = PPO.load(model_dir + "/gen.zip")
    for _ in range(10):
        verify_policy(env, algo, deterministic=False)
