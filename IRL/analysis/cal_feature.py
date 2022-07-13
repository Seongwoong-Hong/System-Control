import os
import json
import pickle
import torch as th
import numpy as np

from common.util import make_env, CPU_Unpickler
from common.wrappers import RewardInputNormalizeWrapper
from algos.tabular.viter import FiniteSoftQiter, FiniteViter

from scipy import io
from copy import deepcopy
import matplotlib.pyplot as plt

demo_path = os.path.abspath(os.path.join("..", "demos"))
log_path = os.path.abspath(os.path.join("..", "tmp", "log"))
with open(f"{demo_path}/bound_info.json", "r") as f:
    bound_info = json.load(f)

def cal_feature_from_data():

    def feature_fn(x):
        x1, x2, x3, x4, a1, a2 = th.split(x, 1, dim=-1)
        out = x ** 2
        ob_sec, act_sec = 6, 4
        for i in range(1, ob_sec):
            out = th.cat([out, (x1 - i / ob_sec) ** 2, (x2 - i / ob_sec) ** 2, (x1 + i / ob_sec) ** 2, (x2 + i / ob_sec) ** 2], dim=1)
        for i in range(1, act_sec):
            out = th.cat([out, (a1 - i / act_sec) ** 2, (a2 - i / act_sec) ** 2, (a1 + i / act_sec) ** 2, (a2 + i / act_sec) ** 2], dim=1)
        return out
        # return th.cat([x, x ** 2], dim=1)
        # return x
        # return x ** 2
        # x1, x2, x3, x4, a1, a2 = th.split(x, 1, dim=1)
        # return th.cat((x, x1*x2, x3*x4, x1*x3, x2*x4, x1*x4, x2*x3, a1*a2, x**2, x**3), dim=1)

    expt = f"19191919/{subj}_{actuation}"
    with open(f"{demo_path}/DiscretizedHuman/{expt}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    # with open(f"{log_path}/DiscretizedHuman/MaxEntIRL/sq_lrdecay_{expt}_{trial}/model/reward_net.pkl", "rb") as f:
    #     reward_fn = CPU_Unpickler(f).load().eval()
    # reward_fn.feature_fn = feature_fn
    # def feature_fn(x):
    #     for layer in reward_fn.layers[:-1]:
    #         x = layer(x)
    #     return x.detach().cpu()
    perturbation = actuation - 1
    bound_dict = bound_info[subj][perturbation]
    normalizer = np.max(np.append(np.append(bound_dict["max_states"], bound_dict["max_torques"])[None, :],
                                  abs(np.append(bound_dict["min_states"], bound_dict["min_torques"]))[None, :],
                                  axis=0),
                        axis=0, keepdims=True)


    features = []
    for traj in expt_trajs:
        inp = np.append(traj.obs[:-1], traj.acts, axis=1) / normalizer
        # inp = traj.obs[:-1] / normalizer
        features.append(np.sum(feature_fn(th.from_numpy(inp).float()).numpy(), axis=0))

    print(f"target mean feature: {np.mean(features, axis=0)}")
    return np.mean(features, axis=0)


def cal_feature_of_learner():

    def feature_fn(x):
        x1, x2, x3, x4, a1, a2 = th.split(x, 1, dim=-1)
        out = x ** 2
        ob_sec, act_sec = 6, 4
        for i in range(1, ob_sec):
            out = th.cat([out, (x1 - i / ob_sec) ** 2, (x2 - i / ob_sec) ** 2, (x1 + i / ob_sec) ** 2, (x2 + i / ob_sec) ** 2], dim=1)
        for i in range(1, act_sec):
            out = th.cat([out, (a1 - i / act_sec) ** 2, (a2 - i / act_sec) ** 2, (a1 + i / act_sec) ** 2, (a2 + i / act_sec) ** 2], dim=1)
        return out
        # return th.cat([x, x ** 2], dim=1)
        # return x
        # return x ** 2
        # x1, x2, x3, x4, a1, a2 = th.split(x, 1, dim=1)
        # return th.cat((x, x ** 2, x1 * x2, x3 * x4, a1 * a2), dim=1)

    device = 'cuda:1'
    bsp = io.loadmat(f"{demo_path}/HPC/{subj}/{subj}i1.mat")['bsp']
    load_dir = f"{log_path}/DiscretizedHuman/MaxEntIRL/sqmany_longlearn_001alpha_lrdecay_19191919"
    with open(f"{load_dir}/{subj}_{actuation}_{trial}/model/reward_net.pkl", "rb") as f:
        reward_fn = CPU_Unpickler(f).load().eval().to(device)
    with open(f"{load_dir}/{subj}_{actuation}_{trial}/{subj}_{actuation}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    init_states = []
    for traj in expt_trajs:
        init_states.append(traj.obs[0])

    reward_fn.feature_fn = feature_fn
    reward_fn.layers[0].weight = th.nn.Parameter(reward_fn.layers[0].weight.detach() / reward_fn.layers[0].weight.norm())

    perturbation = actuation - 1
    bound_dict = bound_info[subj][perturbation]
    normalizer = np.max(np.append(np.append(bound_dict["max_states"], bound_dict["max_torques"])[None, :],
                                  abs(np.append(bound_dict["min_states"], bound_dict["min_torques"]))[None, :],
                                  axis=0),
                        axis=0, keepdims=True)

    env = make_env("DiscretizedHuman-v0", num_envs=1, bsp=bsp, N=[19, 19, 19, 19], NT=[11, 11],
                   init_states=init_states, wrapper=RewardInputNormalizeWrapper, wrapper_kwrags={'rwfn': reward_fn}, )
    perturbation = actuation - 1
    max_states = bound_info[subj][perturbation]["max_states"]
    min_states = bound_info[subj][perturbation]["min_states"]
    max_torques = bound_info[subj][perturbation]["max_torques"]
    min_torques = bound_info[subj][perturbation]["min_torques"]
    env.env_method("set_bounds", max_states, min_states, max_torques, min_torques)
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

    # def feature_fn(x):
    #     for layer in reward_fn.layers[:-1]:
    #         x = layer(x)
    #     return x.detach().cpu()

    # feat_mat = feature_fn(s_vec / normalizer)
    feat_mat = []
    for acts in a_vec:
        inp = th.from_numpy(np.append(s_vec, np.repeat(acts[None, :], len(s_vec), axis=0), axis=1) / normalizer).float().to(device)
        feat_mat.append(feature_fn(inp).cpu().numpy())

    # mean_features = np.sum(Dc.cpu().numpy()[..., None] * np.array(feat_mat), axis=0)
    mean_features = np.sum(np.sum(Dc.cpu().numpy()[..., None] * np.array(feat_mat), axis=0), axis=0)

    print(f"learned mean feature: {mean_features}")
    return mean_features


if __name__ == "__main__":
    # x = [r"$\theta_1$", r"$\theta_2$", r"$\dot{\theta_1}$", r"$\dot{\theta_2}$", "$T_1$", "$T_2$"]
    x = [i for i in range(1, 39)]
    for subj in [f"sub{i:02d}" for i in [5]]:
        for actuation in range(4, 5):
            plt.figure(figsize=[4, 4], dpi=100.0)
            features = []
            for trial in [1]:
                cal_feature_from_data()
                features.append(cal_feature_of_learner())
                # cal_feature_of_learner()
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            print(mean)
            plt.errorbar(x, mean, std, fmt='o')
            plt.scatter(x, cal_feature_from_data(), c='r')
            plt.ylim([2, 6])
            plt.xlabel("states", labelpad=15.0, fontsize=14)
            plt.ylabel("mean feature", labelpad=15.0, fontsize=14)
            plt.tick_params(axis='x', which='major', labelsize=14)
            plt.tick_params(axis='y', which='major', labelsize=10)
            plt.tight_layout()
            plt.show()
