import os
import json
import pickle
import torch as th
import numpy as np

from scipy import io, signal
from imitation.data.rollout import make_sample_until, generate_trajectories
import matplotlib.pyplot as plt

from common.util import make_env, CPU_Unpickler
from common.wrappers import *
from IRL.src import *

demo_path = os.path.abspath(os.path.join("..", "demos"))
log_path = os.path.abspath(os.path.join("..", "tmp", "log"))
with open(f"{demo_path}/bound_info.json", "r") as f:
    bound_info = json.load(f)


def cal_feature_from_data():
    features = []
    for traj in expt_trajs:
        obs_ft = np.sum(feature_fn(th.from_numpy(traj.obs[:-1]).float()).numpy(), axis=0)
        acts_ft = np.sum(feature_fn(th.from_numpy(traj.acts[:-1]).float()).numpy(), axis=0)
        features.append(np.append(obs_ft, acts_ft))

    print(f"target mean feature: {np.mean(features, axis=0)}")
    return np.mean(features, axis=0)


def cal_feature_of_learner():

    init_states = []
    pltqs = []
    for traj in expt_trajs:
        init_states.append(traj.obs[0])
        pltqs.append(traj.pltq)

    reward_fn.feature_fn = feature_fn

    venv = make_env("IP_HPC-v2", num_envs=1, bsp=bsp, pltqs=pltqs,
                    init_states=init_states, wrapper=RewardWrapper, wrapper_kwargs={'rwfn': reward_fn}, )

    agent = IPLQRPolicy(venv, gamma=1, alpha=0.001, device='cpu', verbose=False)
    agent.learn(0)
    sample_until = make_sample_until(n_timesteps=None, n_episodes=75)
    agent_trajs = generate_trajectories(agent, venv, sample_until, deterministic_policy=False)

    features = []
    for traj in agent_trajs:
        obs_ft = np.sum(feature_fn(th.from_numpy(traj.obs[:-1]).float()).numpy(), axis=0)
        acts_ft = np.sum(feature_fn(th.from_numpy(traj.acts[:-1]).float()).numpy(), axis=0)
        features.append(np.append(obs_ft, acts_ft))

    print(f"learned mean feature: {np.mean(features, axis=0)}")
    return np.mean(features, axis=0)


if __name__ == "__main__":
    def feature_fn(x):
        return x ** 2
        # return th.cat([x ** 2, x ** 4], dim=-1)
    x = [r"$\theta_1$", r"$\theta_2$", r"$\dot{\theta_1}$", r"$\dot{\theta_2}$", "$T_1$", "$T_2$"]
    fig = plt.figure(figsize=[8, 4], dpi=100.0)
    for subj in [f"sub{i:02d}" for i in [5]]:
        for actuation in range(1, 2):
            features = []
            for trial in [1]:
                bsp = io.loadmat(f"{demo_path}/HPC/{subj}_single_full/{subj}i1.mat")['bsp']
                expt = f"single_full/{subj}_{actuation}"
                load_dir = f"{log_path}/IP_HPC/MaxEntIRL/sq_001alpha_{expt}"
                with open(f"{load_dir}_{trial}/model/reward_net.pkl", "rb") as f:
                    reward_fn = CPU_Unpickler(f).load().eval().to('cpu')
                with open(f"{load_dir}_{trial}/{subj}_{actuation}.pkl", "rb") as f:
                    expt_trajs = pickle.load(f)
                af = cal_feature_of_learner()
                ef = cal_feature_from_data()
                grad = 2 * reward_fn.reward_layer.weight.detach().numpy() * (ef - af)
                print(np.linalg.norm(grad))
                # cal_feature_of_learner()
            # mean = np.mean(features, axis=0)
            # std = np.std(features, axis=0)
            # print(mean)
            # plt.errorbar(x, mean, std, fmt='o')
            # plt.scatter(x, cal_feature_from_data(), c='r')
            # plt.ylim([2, 6])
            # plt.xlabel("states", labelpad=15.0, fontsize=14)
            # plt.ylabel("mean feature", labelpad=15.0, fontsize=14)
            # plt.tick_params(axis='x', which='major', labelsize=14)
            # plt.tick_params(axis='y', which='major', labelsize=10)
            # plt.tight_layout()
            # plt.show()
