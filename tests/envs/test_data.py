import os
import pickle
import numpy as np

from IRL.scripts.project_policies import def_policy
from common.util import make_env
from common.wrappers import *

from imitation.data import rollout
from matplotlib import pyplot as plt
from scipy import io


def test_hpc_data():
    subi = 1
    sub = f"sub{subi:02d}"
    file = "../../IRL/demos/HPC/" + sub + "/" + sub + f"i{0 + 1}.mat"
    data = {'state': io.loadmat(file)['state'],
            'T': io.loadmat(file)['tq'],
            'pltq': io.loadmat(file)['pltq'],
            'bsp': io.loadmat(file)['bsp'],
            }
    plt.plot(data['state'][:, :2])
    plt.show()
    pltqs = [data['pltq']]
    env = make_env("HPC_custom-v0", use_vec_env=False, n_steps=600, pltqs=pltqs)
    algo = def_policy("HPC", env)
    obs_list = []
    obs = env.reset()
    obs_list.append(obs)
    done = False
    while not done:
        act, _ = algo.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(act)
        obs_list.append(obs)
    plt.plot(np.array(obs_list)[:, :2])
    plt.show()
    print(data['state'][0, :4])


def test_drawing_human_data():
    for subj in [f"sub{i:02d}" for i in [1, 2, 4, 5, 6, 7, 9, 10]]:
        for actu in range(1, 7):
            for exp_trial in range(1, 6):
                for part in range(3):
                    file = f"../../IRL/demos/HPC/{subj}_half/{subj}i{5 * (actu - 1) + exp_trial}_{part}.mat"
                    plt.plot(-io.loadmat(file)['state'][:, 4])
    plt.show()


def test_drawing_pkl_data():
    for subj in [f"sub{i:02d}" for i in [1, 2, 4, 5, 6, 7, 9, 10]]:
        for actu in range(1, 7):
            expert_dir = os.path.join("../../IRL", "demos", "DiscretizedHuman", "17171719", f"{subj}_{actu}.pkl")
            with open(expert_dir, "rb") as f:
                expert_trajs = pickle.load(f)
            for traj in expert_trajs:
                plt.plot(traj.obs[:, 1])
    plt.show()


def test_stepping_pkl_data():
    expert_dir = os.path.join("../../IRL", "demos", "DiscretizedHuman", "19171717_done", "sub06_1.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    bsp = io.loadmat("../../IRL/demos/HPC/sub06/sub06i1.mat")['bsp']
    env = make_env("DiscretizedHuman-v2", bsp=bsp, N=[19, 17, 17, 17], NT=[11, 11], wrapper=DiscretizeWrapper)
    for traj in expert_trajs:
        obs_list = []
        env.reset()
        env.set_state(traj.obs[0])
        obs_list.append(traj.obs[0])
        for act in traj.acts:
            ob, _, _, _ = env.step(act)
            obs_list.append(ob)
        print(f"obs difference: {np.mean(np.abs(traj.obs - np.array(obs_list)))}")
        plt.plot(traj.obs[:, 1], color='k')
        plt.plot(np.array(obs_list)[:, 1], color='b')
    plt.show()
