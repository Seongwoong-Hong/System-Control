import os
import pickle
import numpy as np
import torch as th

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
