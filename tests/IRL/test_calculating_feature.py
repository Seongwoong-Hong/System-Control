import os
import pickle
import numpy as np
import torch as th
from scipy import io

from common.util import CPU_Unpickler

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


def test_calculating_feature_from_pkldata():
    def feature_fn(x):
        return x ** 2

    load_dir = f"{irl_path}/demos/DiscretizedHuman/17171719_log1017"
    with open(f"{load_dir}/sub06_1.pkl", "rb") as f:
        expt_trajs = pickle.load(f)

    features = []
    for traj in expt_trajs:
        inp = np.append(traj.obs[:-1], traj.acts, axis=1)
        features.append(np.sum(feature_fn(inp), axis=0))

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