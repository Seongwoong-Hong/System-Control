import os
import pickle
import torch as th
import numpy as np

from common.util import make_env
from algos.tabular.viter import FiniteSoftQiter

from scipy import io
from copy import deepcopy

demo_path = os.path.abspath(os.path.join("..", "demos"))


def cal_feature_from_data(subj, actuation):

    def feature_fn(x):
        return x ** 2

    with open(f"{demo_path}/DiscretizedHuman/17171719_quadcost_many/{subj}_{actuation}.pkl", "rb") as f:
        expt_traj = pickle.load(f)

    features = []
    for traj in expt_traj:
        inp = np.append(traj.obs[:-1], traj.acts, axis=1)
        features.append(np.sum(feature_fn(inp), axis=0))

    print(f"mean feature: {np.mean(features, axis=0)}")


def cal_feature_from_real_reward(subj, _):

    def feature_fn(x):
        return x ** 2
    bsp = io.loadmat(f"{demo_path}/HPC/{subj}/{subj}i1.mat")['bsp']
    env = make_env("DiscretizedHuman-v2", num_envs=1, bsp=bsp, N=[17, 17, 17, 19], NT=[11, 11])
    agent = FiniteSoftQiter(env, gamma=1, alpha=0.01, device='cpu')
    agent.learn(0)

    D_prev = th.ones(17 * 17 * 17 * 19) / (17 * 17 * 17 * 19)
    Dc = D_prev
    Dc = Dc[None, :] * agent.policy.policy_table[0]
    for t in range(1, 50):
        D = th.zeros_like(D_prev)
        for a in range(agent.policy.act_size):
            D += agent.transition_mat[a] @ (D_prev * agent.policy.policy_table[t - 1, a])
        Dc += agent.policy.policy_table[t] * D[None, :] * agent.gamma ** t
        D_prev = deepcopy(D)

    s_vec, a_vec = env.env_method("get_vectorized")[0]
    feat_mat = []
    for acts in a_vec:
        feat_mat.append(feature_fn(np.append(s_vec, np.repeat(acts[None, :], len(s_vec), axis=0), axis=1)))

    mean_features = np.sum(np.sum(Dc.numpy()[..., None] * np.array(feat_mat), axis=0), axis=0)

    print(f"real mean feature: {mean_features}")


if __name__ == "__main__":
    for subj in [f"sub{i:02d}" for i in [6]]:
        for actuation in range(1, 2):
            cal_feature_from_real_reward(subj, actuation)