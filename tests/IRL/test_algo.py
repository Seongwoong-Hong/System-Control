import os
import time
import pickle
import pytest
from scipy import io

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import verify_policy
from common.wrappers import *

from imitation.algorithms import bc

subj = "sub01"
irl_path = os.path.abspath(os.path.join("..", "..", "IRL"))
bsp = io.loadmat(f"{irl_path}/demos/HPC/{subj}/{subj}i1.mat")['bsp']


@pytest.fixture
def pltqs():
    pltqs = []
    for i in range(5):
        for j in range(5):
            file = os.path.join(irl_path, "demos", "HPC", subj + "_cropped", f"{subj}i{i + 1}_{j}.mat")
            pltqs += [io.loadmat(file)['pltq']]
    return pltqs


@pytest.fixture
def init_states():
    init_states = []
    for i in range(5):
        for j in range(5):
            file = os.path.join(irl_path, "demos", "HPC", subj + "_cropped", f"{subj}i{i + 1}_{j}.mat")
            init_states += [io.loadmat(file)['state'][0, :4]]
    return init_states


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
    name = f"{env_name}/MaxEntIRL/cnn_{subj}_reset_0.1"
    model_dir = os.path.join(irl_path, "tmp", "log", name, "model")
    # algo = bc.reconstruct_policy(model_dir + "/policy")
    algo = SAC.load(model_dir + "/015/agent.pkl")
    a_list, o_list, _ = verify_policy(env, algo, render="human", repeat_num=len(pltqs))


def test_hpc_action_verification(irl_path, pltqs, bsp, subj):
    import matplotlib.pyplot as plt
    import numpy as np
    env_name = "HPC_custom"
    env = make_env(f"{env_name}-v1", wrapper=ActionWrapper, pltqs=pltqs, bsp=bsp)
    name = f"{env_name}/BC/ext_{subj}_noreset"
    model_dir = f"{irl_path}/tmp/log/{name}/model"
    algo = SAC.load(model_dir + "/019/agent.pkl")
    actuations = []
    obs = env.reset()
    done = False
    while not done:
        act, _ = algo.predict(obs, deterministic=True)
        obs, r, done, info = env.step(act)
        actuations.append(info['acts'].reshape(-1))
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


def test_2D(irl_path):
    name = "2DTarget"
    env = make_env(f"{name}-v0")
    model_dir = os.path.join(irl_path, "tmp", "log", name, "GCL", "ext_sac_linear_reset_0.2_2", "model", "029")
    algo = SAC.load(model_dir + f"/agent")
    trajs = []
    for i in range(20):
        st = env.reset()
        done = False
        sts, rs = [], []
        while not done:
            action, _ = algo.predict(st)
            st, r, done, _ = env.step(action)
            sts.append(st)
            rs.append(r)
        trajs.append(np.append(np.array(sts), np.array(rs).reshape(-1, 1), axis=1))
    env.draw(trajs)


def test_1D(irl_path):
    name = "1DTarget"
    env_id = f"{name}_disc"
    env = make_env(f"{env_id}-v0")
    model_dir = os.path.join(irl_path, "tmp", "log", env_id, "MaxEntIRL", "ext_ppo_disc_linear_ppoagent_svm_reset",
                             "model")
    algo = PPO.load(model_dir + "/008/agent")
    a_list, o_list, _ = verify_policy(env, algo, render="None", repeat_num=9, deterministic=False)
    print('end')


def test_discretized_env(init_states):
    from algos.tabular.viter import SoftQiter
    name = "DiscretizedHuman"
    env_id = f"{name}"
    with open(f"{irl_path}/demos/DiscretizedHuman/{subj}_1.pkl", "rb") as f:
        expert = pickle.load(f)
    init_states = []
    for traj in expert:
        init_states += [traj.obs[0]]
    env = make_env(f"{env_id}-v0", num_envs=1, N=[11, 21, 21, 21], bsp=bsp, init_states=init_states)
    model_dir = os.path.join(irl_path, "tmp", "log", "DiscretizedHuman", "MaxEntIRL", f"sq_{subj}_2_finite_action",
                             "model")
    with open(model_dir + "/agent.pkl", "rb") as f:
        agent = pickle.load(f)
    algo = SoftQiter(env, gamma=1, alpha=0.01)
    algo.policy.policy_table = agent.policy.policy_table[0]
    for _ in range(5):
        obs_list = []
        obs = env.reset()
        done = False
        obs_list.append(obs)
        while not done:
            a, _ = algo.predict(obs, deterministic=False)
            ns, _, done, _ = env.step(a)
            env.render()
            time.sleep(env.get_attr("dt")[0])
            obs = ns
            obs_list.append(obs)
        obs_list = np.array(obs_list).reshape(-1, 4)
    env.close()
