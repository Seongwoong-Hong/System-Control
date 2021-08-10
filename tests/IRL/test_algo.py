import os
import pytest
from scipy import io

from IRL.scripts.project_policies import def_policy
from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import verify_policy
from common.wrappers import ActionWrapper

from imitation.algorithms import bc


@pytest.fixture
def irl_path():
    return os.path.abspath(os.path.join("..", "..", "IRL"))


@pytest.fixture
def subj():
    return "sub01"


@pytest.fixture
def pltqs(irl_path, subj):
    pltqs = []
    for i in [0, 5, 10, 15, 20, 25, 30]:
        file = os.path.join(irl_path, "demos", "HPC", subj, f"{subj}i{i + 1}.mat")
        pltqs += [io.loadmat(file)['pltq']]
    return pltqs


@pytest.fixture
def bsp(irl_path, subj):
    return io.loadmat(f"{irl_path}/demos/HPC/{subj}/{subj}i1.mat")['bsp']


def test_hpc_algo(env):
    algo = def_policy("HPC", env)
    a_list, o_list, _ = verify_policy(env, algo)


def test_hpcdiv_algo(tenv):
    algo = def_policy("HPCDiv", tenv)
    for _ in range(35):
        a_list, o_list, _ = verify_policy(tenv, algo)


def test_hpc_learned_policy(irl_path, pltqs, bsp, subj):
    env_name = "HPC_custom"
    env = make_env(f"{env_name}-v0", wrapper=ActionWrapper, pltqs=pltqs, bsp=bsp)
    name = f"{env_name}/BC/extcnn_{subj}_deep_noreset_rewfirst"
    model_dir = os.path.join(irl_path, "tmp", "log", name, "model")
    # algo = bc.reconstruct_policy(model_dir + "/policy")
    algo = SAC.load(model_dir + "/025/agent.zip")
    a_list, o_list, _ = verify_policy(env, algo, render="human", repeat_num=len(pltqs))


def test_hpc_action_verification(irl_path, pltqs, bsp, subj):
    import matplotlib.pyplot as plt
    import numpy as np
    env_name = "HPC_pybullet"
    env = make_env(f"{env_name}-v1", wrapper=ActionWrapper, pltqs=pltqs, bsp=bsp)
    name = f"{env_name}/BC/ext_{subj}_deep_noreset_rewfirst"
    model_dir = f"{irl_path}/tmp/log/{name}/model"
    algo = SAC.load(model_dir + "/036/agent.zip")
    actuations = []
    obs = env.reset()
    done = False
    while not done:
        act, _ = algo.predict(obs, deterministic=True)
        obs, r, done, info = env.step(act)
        actuations.append(info['a'])
    plt.plot(np.array(actuations))
    plt.show()


def test_irl_learned_policy(irl_path):
    env_type = "IDP_custom"
    env = make_env(f"{env_type}-v0", use_vec_env=False)
    name = f"{env_type}/BC/cnn_lqr_ppo_deep_noreset_rewfirst"
    model_dir = os.path.join(irl_path, "tmp", "log", name, "model", "016")
    algo = SAC.load(model_dir + "/agent")
    a_list, o_list, _ = verify_policy(env, algo)


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
