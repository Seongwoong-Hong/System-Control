import os

from algo.torch.ppo import PPO
from algo.torch.sac import SAC
from IRL.scripts.project_policies import def_policy
from common.verification import verify_policy
from common.util import make_env


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


def test_rl_learned_policy():
    env_type = "IP"
    env = make_env(f"{env_type}_custom-v2", n_steps=600)
    name = f"{env_type}_custom/sac"
    model_dir = os.path.join("..", "..", "RL", env_type, "tmp", "log", name, "policies_1")
    algo = SAC.load(model_dir + "/000001000000/model.pkl")
    a_list, o_list, _ = verify_policy(env, algo)


def test_irl_learned_policy():
    env = make_env("IP_custom-v2", n_steps=600, use_vec_env=False)
    name = "IP/ppo/IP_custom2/" + "6"
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    algo = PPO.load(model_dir + "/gen.zip")
    a_list, o_list, _ = verify_policy(env, algo)


def test_idp_policy():
    env = make_env("IDP_custom-v2", n_steps=600, use_vec_env=False)
    algo = def_policy("IDP", env)
    _, _, _ = verify_policy(env, algo)


def test_mujoco_policy():
    name = "Ant"
    env = make_env(f"{name}-v2", use_vec_env=False)
    # model_dir = os.path.join("..", "tmp", "log", "mujoco_envs", "ppo", name, "15", "model")
    model_dir = os.path.join("..", "..", "RL", "mujoco_envs", "tmp", "log", name, "ppo")
    algo = PPO.load(model_dir + "/ppo0.zip")
    for _ in range(10):
        verify_policy(env, algo, deterministic=False)
