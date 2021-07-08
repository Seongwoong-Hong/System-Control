import os
import pytest
from scipy import io

from IRL.scripts.project_policies import def_policy
from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import verify_policy

from imitation.algorithms import bc


@pytest.fixture
def irl_path():
    return os.path.abspath(os.path.join("..", "..", "IRL"))


@pytest.fixture
def pltqs(irl_path):
    pltqs = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        file = os.path.join(irl_path, "demos", "HPC", "sub01", f"sub01i{i + 1}.mat")
        pltqs += [io.loadmat(file)['pltq']]
    return pltqs


def test_hpc_algo(env):
    algo = def_policy("HPC", env)
    a_list, o_list, _ = verify_policy(env, algo)


def test_hpcdiv_algo(tenv):
    algo = def_policy("HPCDiv", tenv)
    for _ in range(35):
        a_list, o_list, _ = verify_policy(tenv, algo)


def test_hpc_learned_policy(irl_path, pltqs):
    env = make_env("HPC_custom-v0", pltqs=pltqs)
    name = "HPC_custom/BC/test"
    model_dir = os.path.join(irl_path, "tmp", "log", name, "model")
    algo = bc.reconstruct_policy(model_dir + "/policy")
    for _ in range(10):
        a_list, o_list, _ = verify_policy(env, algo)


def test_irl_learned_policy(irl_path):
    env_type = "IDP_custom"
    env = make_env(f"{env_type}-v1", use_vec_env=False)
    name = f"{env_type}/BC/test"
    model_dir = os.path.join(irl_path, "tmp", "log", name, "model")
    algo = bc.reconstruct_policy(model_dir + "/policy")
    a_list, o_list, _ = verify_policy(env, algo, deterministic=True)


def test_idp_policy():
    env = make_env("IDP_custom-v0", use_vec_env=False)
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
