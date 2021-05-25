import os

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
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
    name = "HPC/MaxEntIRL/HPC_customtest/" + ""
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    algo = SAC.load(model_dir + "/agent.zip")
    for _ in range(10):
        a_list, o_list, _ = verify_policy(env, algo)


def test_irl_learned_policy():
    env_type = "IDP"
    env = make_env(f"{env_type}_custom-v2", n_steps=600, use_vec_env=False)
    name = f"{env_type}/MaxEntIRL/{env_type}_custom_fast"
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    algo = SAC.load(model_dir + "/agent")
    a_list, o_list, _ = verify_policy(env, algo)


def test_idp_policy():
    env = make_env("IDP_custom-v2", n_steps=600, use_vec_env=False)
    algo = def_policy("IDP", env)
    _, _, _ = verify_policy(env, algo)


def test_mujoco_policy():
    name = "Hopper"
    env = make_env(f"{name}-v2", use_vec_env=False)
    model_dir = os.path.join("..", "tmp", "log", "mujoco_envs", "ppo", name, "10", "model")
    # model_dir = os.path.join("..", "..", "RL", "mujoco_envs", "tmp", "log", name, "ppo")
    algo = PPO.load(model_dir + "/gen")
    for _ in range(10):
        verify_policy(env, algo, deterministic=False)
