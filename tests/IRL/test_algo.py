import os
import time
import json
import pickle
import pytest
import torch as th
import numpy as np
from scipy import io
from matplotlib import pyplot as plt

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from algos.tabular.viter import FiniteSoftQiter, SoftQiter
from common.util import make_env, CPU_Unpickler
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


def test_irl_learned_policy():
    env_type = "2DTarget"
    with open(f"{irl_path}/demos/{env_type}/sac_1.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    init_states = []
    for traj in expert_trajs:
        init_states.append(traj.obs[0])
    env = make_env(f"{env_type}-v0", init_states=init_states)
    name = f"{env_type}/GCL/sq_sac_1_sac_resetstd"
    model_dir = os.path.join(irl_path, "tmp", "log", name, "model")
    algo = SAC.load(model_dir + "/agent")
    for _ in range(len(init_states)):
        ob = env.reset()
        env.render()
        done = False
        while not done:
            act, _ = algo.predict(ob, deterministic=False)
            ob, _, done, _ = env.step(act)
            env.render()
            time.sleep(env.dt)


def test_mujoco_policy():
    name = "Hopper"
    env = make_env(f"{name}-v2", use_vec_env=False)
    model_dir = os.path.join(irl_path, "tmp", "log", "mujoco_envs", "ppo", name, "1", "model")
    # model_dir = os.path.join("..", "..", "RL", "mujoco_envs", "tmp", "log", name, "ppo")
    algo = PPO.load(model_dir + "/gen")
    for _ in range(10):
        verify_policy(env, algo, deterministic=False)


def test_2D():
    name = "2DTarget"
    env = make_env(f"{name}-v2")
    model_dir = os.path.join(irl_path, "tmp", "log", name, "GCL", "sq_sac_1_sac", "model")
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


@pytest.mark.parametrize("trial", range(1, 5))
def test_discretized_env(trial):
    def feature_fn(x):
        # x1, x2, a1, a2 = th.split(x, 1, dim=-1)
        # out = x ** 2
        # for i in range(1, 50):
        #     out = th.cat([out, (x1 - i / 49) ** 2, (x2 - i / 49) ** 2], dim=1)
        # for i in range(1, 7):
        #     out = th.cat([out, (a1 - i / 6) ** 2, (a2 - i / 6) ** 2], dim=1)
        # return out
        return th.cat([x, x**2], dim=1)
        # return th.cat([x, x**2, x**3, x**4], dim=1)
    env_type = "DiscretizedPendulum"
    name = f"{env_type}"
    expt = "2929_51_lqr/quadcost_from_contlqr_many"
    with open(f"{irl_path}/demos/{env_type}/{expt}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    init_states = []
    for traj in expt_trajs:
        init_states += [traj.obs[0]]
    rwfn_dir = irl_path + f"/tmp/log/{name}/MaxEntIRL/ext_01alpha_{expt}_{trial}/model"
    with open(rwfn_dir + "/reward_net.pkl", "rb") as f:
        rwfn = CPU_Unpickler(f).load().eval().to('cpu')
    rwfn.feature_fn = feature_fn
    # env = make_env(f"{env_id}-v0", num_envs=1, N=[19, 19, 19, 19], bsp=bsp, init_states=init_states)
    env = make_env(f"{name}-v2", num_envs=1, N=[29, 29], NT=[51], wrapper=RewardInputNormalizeWrapper, wrapper_kwargs={'rwfn': rwfn})
    # env = make_env(f"{name}-v2", num_envs=1, wrapper=DiscretizeWrapper)
    eval_env = make_env(f"{name}-v0", N=[29, 29], NT=[51], init_states=init_states, wrapper=DiscretizeWrapper)
    agent = FiniteSoftQiter(env, gamma=1, alpha=0.01, device=rwfn.device, verbose=False)
    agent.learn(0)
    agent.set_env(eval_env)

    fig1 = plt.figure(figsize=[9, 14.4])
    fig2 = plt.figure(figsize=[9.6, 4.8])
    ax2_ex = fig2.add_subplot(1, 2, 1)
    ax2_ag = fig2.add_subplot(1, 2, 2)
    ax11 = fig1.add_subplot(6, 2, 1)
    ax12 = fig1.add_subplot(6, 2, 2)
    ax21 = fig1.add_subplot(6, 2, 3)
    ax22 = fig1.add_subplot(6, 2, 4)
    ax51 = fig1.add_subplot(6, 2, 9)
    ax52 = fig1.add_subplot(6, 2, 10)
    ax61 = fig1.add_subplot(6, 2, 11)
    ax62 = fig1.add_subplot(6, 2, 12)

    for traj in expt_trajs:
        # Expert Trajectories
        # ax2_ex.plot(traj.obs[:-1, 0], traj.obs[:-1, 1])
        # ax2_ex.set_xlim([eval_env.obs_low[0], eval_env.obs_high[0]])
        # ax2_ex.set_ylim([eval_env.obs_low[1], eval_env.obs_high[1]])
        ax11.plot(traj.obs[:-1, 0])
        ax21.plot(traj.obs[:-1, 1])
        ax51.plot(traj.acts[:, 0])
        # ax61.plot(traj.acts[:, 1])
        # ax11.set_ylim([eval_env.obs_low[0], eval_env.obs_high[0]])
        # ax21.set_ylim([eval_env.obs_low[1], eval_env.obs_high[1]])
        # ax51.set_ylim([eval_env.acts_low[0], eval_env.acts_high[0]])
        # ax61.set_ylim([eval_env.acts_low[1], eval_env.acts_high[1]])

    for i in range(len(expt_trajs)):
        # obs, acts = [], []
        eval_env.reset()
        ob = init_states[i]
        # obs.append(ob)
        # done = False
        # while not done:
        #     act, _ = agent.predict(ob, deterministic=False)
        #     ob, _, done, _ = eval_env.step(act[0])
        #     acts.append(act[0])
        #     obs.append(ob)
        # obs = np.array(obs)
        # acts = np.array(acts)
        obs, acts, _ = agent.predict(ob, deterministic=False)
        # ax2_ag.plot(obs[:-1, 0], obs[:-1, 1])
        # ax2_ag.set_xlim([eval_env.obs_low[0], eval_env.obs_high[0]])
        # ax2_ag.set_ylim([eval_env.obs_low[1], eval_env.obs_high[1]])
        ax12.plot(obs[:-1, 0])
        ax22.plot(obs[:-1, 1])
        ax52.plot(acts[:, 0])
        # ax62.plot(acts[:, 1])
        # ax12.set_ylim([eval_env.obs_low[0], eval_env.obs_high[0]])
        # ax22.set_ylim([eval_env.obs_low[1], eval_env.obs_high[1]])
        # ax52.set_ylim([eval_env.acts_low[0], eval_env.acts_high[0]])
        # ax62.set_ylim([eval_env.acts_low[1], eval_env.acts_high[1]])
        # done = False
        # obs_list.append(obs)
        # while not done:
        #     a, _ = algo.predict(obs, deterministic=False)
        #     ns, _, done, _ = eval_env.step(a)
        #     eval_env.render()
        #     time.sleep(env.get_attr("dt")[0])
        #     obs = ns
        #     obs_list.append(obs)
        # obs_list = np.array(obs_list).reshape(-1, 4)
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()
    env.close()

params = []
for actu in range(4, 5):
    for trial in range(1, 3):
        params.append([actu, trial])


@pytest.mark.parametrize("actu, trial", params)
def test_learned_human_policy(actu, trial):

    def feature_fn(x):
        # x1, x2, x3, x4, a1, a2 = th.split(x, 1, dim=-1)
        # out = x ** 2
        # ob_sec, act_sec = 4, 3
        # for i in range(1, ob_sec):
        #     out = th.cat([out, (x1 - i / ob_sec) ** 2, (x2 - i / ob_sec) ** 2, (x3 - i /ob_sec) ** 2, (x4 - i / ob_sec) ** 2,
        #                   (x1 + i / ob_sec) ** 2, (x2 + i / ob_sec) ** 2, (x3 + i / ob_sec) ** 2, (x4 + i / ob_sec) ** 2], dim=1)
        # for i in range(1, act_sec):
        #     out = th.cat([out, (a1 - i / act_sec) ** 2, (a2 - i / act_sec) ** 2, (a1 + i / act_sec) ** 2, (a2 + i / act_sec) ** 2], dim=1)
        # return out
        # return x
        # return x ** 2
        return th.cat([x, x ** 2], dim=1)
        # x1, x2, x3, x4, a1, a2 = th.split(x, 1, dim=1)
        # return th.cat((x, x ** 2, x1 * x2, x3 * x4, a1 * a2), dim=1)
    env_type = "DiscretizedPendulum"
    name = f"{env_type}"
    subj = "sub05"
    device = 'cpu'
    # expt = f"19191919/{subj}_{actu}"
    expt = f"301201_101_lqr/quadcost_from_contlqr"
    with open(irl_path + f"/demos/{env_type}/{expt}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    with open(irl_path + f"/demos/bound_info.json", "r") as f:
        bound_info = json.load(f)
    init_states = []
    for traj in expt_trajs:
        init_states.append(traj.obs[0])
    bsp = io.loadmat(os.path.join(irl_path, "demos", "HPC", subj, subj + "i1.mat"))['bsp']
    rwfn_dir = irl_path + f"/tmp/log/{name}/MaxEntIRL/ext_01alpha_{expt}_{trial}/model"
    with open(rwfn_dir + "/reward_net.pkl", "rb") as f:
        rwfn = CPU_Unpickler(f).load().to(device)
    rwfn.feature_fn = feature_fn
    # rwfn.layers[0].weight = th.nn.Parameter(rwfn.layers[0].weight.detach() / 5)
    # rwfn.layers[0].weight = th.nn.Parameter(th.tensor([[-0.03946868, -0.24135341, -0.02744996, -0.0490342 , -0.07611195, -0.04075731]]))

    # env = make_env(f"{name}-v2", num_envs=1, wrapper=RewardInputNormalizeWrapper, wrapper_kwargs={'rwfn':rwfn})
    # eval_env = make_env(f"{name}-v0", num_envs=1, wrapper=DiscretizeWrapper)
    env = make_env(f"{name}-v2", num_envs=1, N=[301, 201], NT=[101],
                   wrapper=RewardInputNormalizeWrapper, wrapper_kwargs={'rwfn': rwfn})
    eval_env = make_env(f"{name}-v0", N=[301, 201], NT=[101], wrapper=DiscretizeWrapper,
                        bsp=bsp, init_states=init_states,)
    # perturbation = actu - 1
    # max_states = bound_info[subj][perturbation]["max_states"]
    # min_states = bound_info[subj][perturbation]["min_states"]
    # max_torques = bound_info[subj][perturbation]["max_torques"]
    # min_torques = bound_info[subj][perturbation]["min_torques"]
    # env.env_method('set_bounds', max_states, min_states, max_torques, min_torques)
    # eval_env.set_bounds(max_states, min_states, max_torques, min_torques)
    agent = FiniteSoftQiter(env=env, gamma=1, alpha=0.01, device=device, verbose=True)
    agent.learn(0)
    agent.set_env(eval_env)
    # agent = SoftQiter(env=env, gamma=1, alpha=0.01, device=device, verbose=True)
    # agent.policy.policy_table = agent2.policy.policy_table[0]

    fig1 = plt.figure(figsize=[9, 14.4])
    ax11 = fig1.add_subplot(6, 2, 1)
    ax12 = fig1.add_subplot(6, 2, 2)
    ax21 = fig1.add_subplot(6, 2, 3)
    ax22 = fig1.add_subplot(6, 2, 4)
    ax31 = fig1.add_subplot(6, 2, 5)
    ax32 = fig1.add_subplot(6, 2, 6)
    ax41 = fig1.add_subplot(6, 2, 7)
    ax42 = fig1.add_subplot(6, 2, 8)
    ax51 = fig1.add_subplot(6, 2, 9)
    ax52 = fig1.add_subplot(6, 2, 10)
    ax61 = fig1.add_subplot(6, 2, 11)
    ax62 = fig1.add_subplot(6, 2, 12)

    for traj in expt_trajs:
        # Expert Trajectories
        ax11.plot(traj.obs[:-1, 0])
        ax21.plot(traj.obs[:-1, 1])
        # ax31.plot(traj.obs[:-1, 2])
        # ax41.plot(traj.obs[:-1, 3])
        ax51.plot(traj.acts[:, 0])
        # ax61.plot(traj.acts[:, 1])
        # ax11.set_ylim([min_states[0], max_states[0]])
        # ax21.set_ylim([min_states[1], max_states[1]])
        # ax31.set_ylim([min_states[2], max_states[2]])
        # ax41.set_ylim([min_states[3], max_states[3]])
        # ax51.set_ylim([min_torques[0], max_torques[0]])
        # ax61.set_ylim([min_torques[1], max_torques[1]])

    # for i in range(len(expt_trajs)):
        # Agent Trajectories
        # obs, acts = [], []
        ob = eval_env.reset()
        # obs.append(ob)
        # done = False
        # while not done:
        #     act, _ = agent.predict(ob, deterministic=False)
        #     ob, _, done, _ = eval_env.step(act[0])
        #     acts.append(act[0])
        #     obs.append(ob)
        # obs = np.array(obs)
        # acts = np.array(acts)

        # ob = init_states[i % len(init_states)]
        obs, acts, _ = agent.predict(ob, deterministic=False)

        ax12.plot(obs[:-1, 0])
        ax22.plot(obs[:-1, 1])
        # ax32.plot(obs[:-1, 2])
        # ax42.plot(obs[:-1, 3])
        ax52.plot(acts[:, 0])
        # ax62.plot(acts[:, 1])
        # ax12.set_ylim([min_states[0], max_states[0]])
        # ax22.set_ylim([min_states[1], max_states[1]])
        # ax32.set_ylim([min_states[2], max_states[2]])
        # ax42.set_ylim([min_states[3], max_states[3]])
        # ax52.set_ylim([min_torques[0], max_torques[0]])
        # ax62.set_ylim([min_torques[1], max_torques[1]])
        eval_env.render()
        for t in range(50):
            obs_idx = eval_env.get_idx_from_obs(ob)
            act_idx = agent.policy.choice_act(agent.policy.policy_table[t].T[obs_idx])
            act = eval_env.get_acts_from_idx(act_idx)
            ob, r, _, _ = eval_env.step(act[0])
            eval_env.render()
            time.sleep(eval_env.dt)
    plt.tight_layout()
    plt.show()
