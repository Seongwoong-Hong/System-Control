import os
import pytest
import pickle
import numpy as np
import torch as th
from scipy import io
from copy import deepcopy

from common.util import CPU_Unpickler, make_env
from common.wrappers import *
from algos.tabular.viter import FiniteSoftQiter

from imitation.data.rollout import flatten_trajectories

irl_path = os.path.abspath(os.path.join("..", "..", "IRL"))
log_path = os.path.join(irl_path, "tmp", "log")


def test_calculating_feature_from_ann():
    def feature_fn(x):
        return x

    load_dir = f"{log_path}/DiscretizedHuman/MaxEntIRL/ann_handnorm_17171719_quadcost_finite"
    with open(f"{load_dir}/sub06_1_1/model/reward_net.pkl", "rb") as f:
        reward_fn = CPU_Unpickler(f).load()
    with open(f"{load_dir}/sub06_1_1/sub06_1.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    reward_fn.feature_fn = feature_fn
    expt_trans = flatten_trajectories(expt_trajs)
    inp = feature_fn(th.from_numpy(np.append(expt_trans.obs, expt_trans.acts, axis=1)).float())
    for layer in reward_fn.layers[:-1]:
        inp = layer(inp)
    print(inp)


@pytest.mark.parametrize("trial", [1])
def test_calculating_feature(trial):

    def feature_fn(x):
        # return x ** 2
        # return th.cat([x, x**2, x**3, x**4], dim=1)
        return th.cat([x, x ** 2], dim=1)

    bsp = io.loadmat(f"{irl_path}/demos/HPC/sub01/sub01i1.mat")['bsp']
    expt_name = "01alpha_nobias_many"
    load_dir = f"{log_path}/SpringBall/MaxEntIRL/ext_01alpha_dot4_2_10/{expt_name}_{trial}"
    with open(f"{load_dir}/model/1000/reward_net.pkl", "rb") as f:
        reward_fn = CPU_Unpickler(f).load().eval()
    with open(f"{load_dir}/{expt_name}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    init_states = []
    for traj in expt_trajs:
        init_states.append(traj.obs[0])
    reward_fn.feature_fn = feature_fn
    # reward_fn.layers[0].weight = th.nn.Parameter(th.tensor([0.2333, 0.0025, -0.0032, -1.1755, -0.1169, -0.0118]))
    # env = make_env("DiscretizedHuman-v0", num_envs=1, bsp=bsp, N=[19, 19, 19, 19], NT=[11, 11],
    #                wrapper=RewardInputNormalizeWrapper, wrapper_kwrags={'rwfn': reward_fn}, init_states=init_states, )
    # env = make_env("SpringBall-v0", init_states=init_states, wrapper=RewardInputNormalizeWrapper, wrapper_kwargs={'rwfn': reward_fn})
    env = make_env("SpringBall-v0", init_states=init_states)#, wrapper=RewardWrapper, wrapper_kwargs={'rwfn': reward_fn})
    agent = FiniteSoftQiter(env, gamma=1, alpha=0.01, device='cpu', verbose=False)
    agent.learn(0)

    D_prev = th.zeros([agent.policy.obs_size], dtype=th.float32)
    init_obs, _ = env.get_init_vector()
    init_idx = env.get_idx_from_obs(init_obs)
    for i in range(len(init_idx)):
        D_prev[init_idx[i]] = (init_idx == init_idx[i]).sum() / len(init_idx)

    Dc = D_prev
    Dc = Dc[None, :] * agent.policy.policy_table[0]
    for t in range(1, 40):
        D = th.zeros_like(D_prev)
        for a in range(agent.policy.act_size):
            D += agent.transition_mat[a] @ (D_prev * agent.policy.policy_table[t - 1, a])
        Dc += agent.policy.policy_table[t] * D[None, :] * agent.gamma ** t
        D_prev = deepcopy(D)

    s_vec, a_vec = env.get_vectorized()

    # def feature_fn(x):
    #     x = th.cat([x, x**2, x**3, x**4], dim=1)
    #     for layer in reward_fn.layers[:-1]:
    #         x = layer(x)
    #     return x.detach().cpu()

    feat_mat = []
    for acts in a_vec:
        feat_mat.append(feature_fn(
            th.from_numpy(
                np.append(s_vec, np.repeat(acts[None, :], len(s_vec), axis=0), axis=1) / np.array([[0.4, 2, 10]])
            ).float()).numpy())
    mean_features = np.sum(np.sum(Dc.cpu().numpy()[..., None] * np.array(feat_mat), axis=0), axis=0)

    print(f"learned mean feature: {mean_features}")


def test_calculating_feature_from_pkldata():
    def feature_fn(x):
        return th.cat([x, x**2], dim=1)
        # return x ** 2

    load_dir = f"{irl_path}/demos/SpringBall/dot4_2_10"
    with open(f"{load_dir}/01alpha_nobias_many.pkl", "rb") as f:
        expt_trajs = pickle.load(f)

    features = []
    for traj in expt_trajs:
        inp = np.append(traj.obs[:-1], traj.acts, axis=1)
        features.append(np.sum(feature_fn(
            th.from_numpy(inp / np.array([[0.4, 2, 10]])).float()
        ).numpy(), axis=0))

    print(f"target mean feature: {np.mean(features, axis=0)}")


def test_calculating_feature_from_data():
    def feature_fn(x):
        return x ** 2

    load_dir = f"{irl_path}/demos/HPC/sub06_half"
    states, Ts = [], []
    for trial in range(1, 6):
        for part in range(3):
            state = -io.loadmat(load_dir + f"/sub06i{trial}_{part}.mat")['state'][:, :4]
            T = -io.loadmat(load_dir + f"/sub06i{trial}_{part}.mat")['tq']
            states.append(np.sum(feature_fn(state), axis=0))
            Ts.append(np.sum(feature_fn(T), axis=0))
    states = np.vstack(states)
    Ts = np.vstack(Ts)
    inp = np.append(states, Ts, axis=1)

    print(f"{np.mean(inp, axis=0)}")