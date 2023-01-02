import os
import json
import pickle

from common.util import make_env
from common.wrappers import *

from matplotlib import pyplot as plt
from scipy import io


def test_drawing_human_data():
    fig1 = plt.figure(figsize=[9.6, 9.6])
    ax11 = fig1.add_subplot(3, 2, 1)
    ax12 = fig1.add_subplot(3, 2, 2)
    ax21 = fig1.add_subplot(3, 2, 3)
    ax22 = fig1.add_subplot(3, 2, 4)
    ax31 = fig1.add_subplot(3, 2, 5)
    ax32 = fig1.add_subplot(3, 2, 6)
    for subj in [f"sub{i:02d}" for i in [5]]:
        for actu in range(1, 2):
            for exp_trial in range(1, 6):
                file = f"../../IRL/demos/HPC/{subj}/{subj}i{5 * (actu - 1) + exp_trial}.mat"
                ax11.plot(-io.loadmat(file)['state'][:, 0])
                ax12.plot(-io.loadmat(file)['state'][:, 1])
                ax21.plot(-io.loadmat(file)['state'][:, 2])
                ax22.plot(-io.loadmat(file)['state'][:, 3])
                ax31.plot(-io.loadmat(file)['tq'][:, 0])
                ax32.plot(-io.loadmat(file)['tq'][:, 1])
                ax11.set_ylim([-.05, .05])
                ax12.set_ylim([-.2, .05])
                ax21.set_ylim([-.18, .3])
                ax22.set_ylim([-.4, .45])
                ax31.set_ylim([-60, 60])
                ax32.set_ylim([-20, 50])
    fig1.tight_layout()
    plt.show()


def test_drawing_hist_of_human_obs():
    obs_stack = []
    for subj in [f"sub{i:02d}" for i in [1, 2, 4, 5, 6, 7, 9, 10]]:
        for actu in range(1, 7):
            for exp_trial in range(1, 6):
                for part in range(3):
                    file = f"../../IRL/demos/HPC/{subj}_half/{subj}i{5 * (actu - 1) + exp_trial}_{part}.mat"
                    obs_stack.append(-io.loadmat(file)['tq'][:, 1])
    sorted_obs = np.sort(np.hstack(obs_stack))
    plt.hist(np.hstack(obs_stack), bins=17, range=(-2.4, 1.3))
    plt.show()


def test_drawing_pkl_data():
    for subj in [f"sub{i:02d}" for i in range(1, 2)]:
        for actu in range(4, 5):
            # bound_dict = bound_info[subj][actu - 1]
            expert_dir = os.path.join("../../IRL", "demos", "HPC", f"full/{subj}_{actu}.pkl")
            # expert_dir = os.path.join("../../IRL", "demos", "SpringBall", "cont", f"quadcost_lqr_many.pkl")
            with open(expert_dir, "rb") as f:
                expert_trajs = pickle.load(f)
            fig1 = plt.figure()
            for i in range(4):
                fig1.add_subplot(4, 1, i + 1)
            for traj in expert_trajs:
                for i in range(2):
                    fig1.axes[i].plot(traj.obs[:-1, i])
                for j in range(1):
                    fig1.axes[j + 2].plot(traj.acts[:, j])
                for k in range(1):
                    fig1.axes[k + 3].plot(traj.pltq[:, k])
            fig1.tight_layout()
            fig1.show()


def test_stepping_pkl_data():
    expert_dir = os.path.join("../../IRL", "demos", "DiscretizedPendulum", "301201_101_lqr", "quadcost_150050_from_contlqr.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    bsp = io.loadmat("../../IRL/demos/HPC/sub06/sub06i1.mat")['bsp']
    env = make_env("DiscretizedPendulum-v2", N=[301, 201], NT=[101], wrapper=DiscretizeWrapper)
    for traj in expert_trajs:
        obs_list = []
        env.reset()
        env.set_state(traj.obs[0])
        obs_list.append(traj.obs[0])
        for act in traj.acts:
            ob, _, _, _ = env.step(act)
            obs_list.append(ob)
        print(f"obs difference: {np.mean(np.abs(traj.obs - np.array(obs_list)))}")
        plt.plot(traj.obs[:, 0], color='k')
        plt.plot(np.array(obs_list)[:, 0], color='b')
    plt.show()
