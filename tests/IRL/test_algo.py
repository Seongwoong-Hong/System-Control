import os
import pytest

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from IRL.scripts.project_policies import def_policy
from common.verification import verify_policy
from common.util import make_env


@pytest.fixture
def irl_path():
    return os.path.abspath(os.path.join("..", "..", "IRL"))


def test_hpc_algo(env):
    algo = def_policy("HPC", env)
    a_list, o_list, _ = verify_policy(env, algo)


def test_hpcdiv_algo(tenv):
    algo = def_policy("HPCDiv", tenv)
    for _ in range(35):
        a_list, o_list, _ = verify_policy(tenv, algo)


def test_hpc_learned_policy(env, irl_path):
    name = "HPC/MaxEntIRL/no_sub01_1&2/" + "add_rew_learning"
    model_dir = os.path.join(irl_path, "tmp", "log", name, "policies_1")
    algo = SAC.load(model_dir + "/000000250000/model.pkl")
    for _ in range(10):
        a_list, o_list, _ = verify_policy(env, algo)


def test_irl_learned_policy(irl_path):
    env_type = "IDP"
    env = make_env(f"{env_type}_custom-v0", n_steps=600, use_vec_env=False)
    name = f"{env_type}/MaxEntIRL/no_lqr"
    model_dir = os.path.join(irl_path, "tmp", "log", name, "model")
    algo = SAC.load(model_dir + "/030/agent")
    a_list, o_list, _ = verify_policy(env, algo, deterministic=True)


def test_idp_policy():
    env = make_env("IDP_custom-v0", n_steps=600, use_vec_env=False)
    algo = def_policy("IDP", env, noise_lv=0.5)
    _, _, _ = verify_policy(env, algo, deterministic=False)


def test_mujoco_policy(irl_path):
    name = "Hopper"
    env = make_env(f"{name}-v2", use_vec_env=False)
    model_dir = os.path.join(irl_path, "tmp", "log", "mujoco_envs", "ppo", name, "10", "model")
    # model_dir = os.path.join("..", "..", "RL", "mujoco_envs", "tmp", "log", name, "ppo")
    algo = PPO.load(model_dir + "/gen")
    for _ in range(10):
        verify_policy(env, algo, deterministic=False)
