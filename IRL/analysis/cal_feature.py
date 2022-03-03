import os
import pickle
import torch as th
import numpy as np

from common.util import make_env, CPU_Unpickler
from common.wrappers import RewardInputNormalizeWrapper
from algos.tabular.viter import FiniteSoftQiter

from scipy import io
from copy import deepcopy

demo_path = os.path.abspath(os.path.join("..", "demos"))
log_path = os.path.abspath(os.path.join("..", "tmp", "log"))
normalizer = np.array([[0.16, 0.7, 0.8, 2.4, 100., 100.]])


def cal_feature_from_data():

    def feature_fn(x):
        return x ** 2

    with open(f"{demo_path}/DiscretizedHuman/17171719_quadcost_finite/{subj}_{actuation}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    load_dir = f"{log_path}/DiscretizedHuman/MaxEntIRL/ann_handnorm_17171719_quadcost_finite"
    with open(f"{load_dir}/{subj}_{actuation}_{trial}/model/reward_net.pkl", "rb") as f:
        reward_fn = CPU_Unpickler(f).load()
    reward_fn.feature_fn = feature_fn

    # def feature_fn(x):
    #     x = th.from_numpy(x).float()
    #     for layer in reward_fn.layers[:-1]:
    #         x = layer(x)
    #     return x.detach().numpy()

    features = []
    for traj in expt_trajs:
        inp = np.append(traj.obs[:-1], traj.acts, axis=1) / normalizer
        # inp = traj.obs[:-1] / normalizer
        features.append(np.sum(feature_fn(inp), axis=0))

    print(f"target mean feature: {np.mean(features, axis=0)}")


def cal_feature_of_learner():

    def feature_fn(x):
        return x ** 2

    device = 'cuda:0'
    bsp = io.loadmat(f"{demo_path}/HPC/{subj}/{subj}i1.mat")['bsp']
    load_dir = f"{log_path}/DiscretizedHuman/MaxEntIRL/sq_handnorm_17171719_quadcost_finite"
    with open(f"{load_dir}/{subj}_{actuation}_{trial}/model/reward_net.pkl", "rb") as f:
        reward_fn = CPU_Unpickler(f).load().eval().to(device)
    with open(f"{load_dir}/{subj}_{actuation}_{trial}/{subj}_{actuation}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    init_states = []
    for traj in expt_trajs:
        init_states.append(traj.obs[0])
    reward_fn.feature_fn = feature_fn
    env = make_env("DiscretizedHuman-v2", num_envs=1, bsp=bsp, N=[17, 17, 17, 19], NT=[11, 11],
                   )#wrapper=RewardInputNormalizeWrapper, wrapper_kwrags={'rwfn': reward_fn})#init_states=init_states)
    agent = FiniteSoftQiter(env, gamma=1, alpha=0.001, device=device, verbose=False)
    agent.learn(0)

    D_prev = th.zeros([agent.policy.obs_size], dtype=th.float32).to(device)
    init_obs, _ = env.env_method("get_init_vector")[0]
    init_idx = env.env_method("get_idx_from_obs", init_obs)[0]
    for i in range(len(init_idx)):
        D_prev[init_idx[i]] = (init_idx == init_idx[i]).sum() / len(init_idx)

    # D_prev = th.ones(agent.policy.obs_size) / agent.policy.obs_size
    Dc = D_prev
    Dc = Dc[None, :] * agent.policy.policy_table[0]
    for t in range(1, 50):
        D = th.zeros_like(D_prev)
        for a in range(agent.policy.act_size):
            D += agent.transition_mat[a] @ (D_prev * agent.policy.policy_table[t - 1, a])
        # Dc += D * agent.gamma ** t
        Dc += agent.policy.policy_table[t] * D[None, :] * agent.gamma ** t
        D_prev = deepcopy(D)

    s_vec, a_vec = env.env_method("get_vectorized")[0]
    # feat_mat = feature_fn(s_vec / normalizer)

    # def feature_fn(x):
    #     x = th.from_numpy(x).float().to(device)
    #     for layer in reward_fn.layers[:-1]:
    #         x = layer(x)
    #     return x.detach().cpu().numpy()

    feat_mat = []
    for acts in a_vec:
        feat_mat.append(feature_fn(np.append(s_vec, np.repeat(acts[None, :], len(s_vec), axis=0), axis=1) / normalizer))

    # mean_features = np.sum(Dc.numpy()[..., None] * np.array(feat_mat), axis=0)
    mean_features = np.sum(np.sum(Dc.cpu().numpy()[..., None] * np.array(feat_mat), axis=0), axis=0)

    print(f"learned mean feature: {mean_features}")


if __name__ == "__main__":
    for subj in [f"sub{i:02d}" for i in [6]]:
        for actuation in [1]:
            for trial in range(1, 2):
                cal_feature_from_data()
                cal_feature_of_learner()