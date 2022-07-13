import os
import json
import pickle

from common.util import make_env
from common.wrappers import *

from matplotlib import pyplot as plt
from scipy import io


def test_drawing_human_data():
    for subj in [f"sub{i:02d}" for i in [1, 2, 4, 5, 6, 7, 9, 10]]:
        for actu in range(1, 7):
            for exp_trial in range(1, 6):
                for part in range(3):
                    file = f"../../IRL/demos/HPC/{subj}_half/{subj}i{5 * (actu - 1) + exp_trial}_{part}.mat"
                    plt.plot(-io.loadmat(file)['state'][:, 4])
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
    with open("../../IRL/demos/bound_info.json", "r") as f:
        bound_info = json.load(f)
    env = make_env("DiscretizedPendulum-v2", N=[201, 201], NT=[101])
    for subj in [f"sub{i:02d}" for i in [5]]:
        for actu in range(4, 5):
            bound_dict = bound_info[subj][actu - 1]
            # expert_dir = os.path.join("../../IRL", "demos", "DiscretizedHuman", "19191919_lqr", f"quadcost_from_contlqr_sub05.pkl")
            expert_dir = os.path.join("../../IRL", "demos", "DiscretizedPendulum", "databased_21_lqr", f"quadcost_lqr.pkl")
            # expert_dir = os.path.join("../../IRL", "demos", "SpringBall", "cont", f"quadcost.pkl")
            with open(expert_dir, "rb") as f:
                expert_trajs = pickle.load(f)
            fig1 = plt.figure(figsize=[9.6, 9.6])
            ax11 = fig1.add_subplot(3, 2, 1)
            ax12 = fig1.add_subplot(3, 2, 2)
            ax21 = fig1.add_subplot(3, 2, 3)
            ax22 = fig1.add_subplot(3, 2, 4)
            ax31 = fig1.add_subplot(3, 2, 5)
            ax32 = fig1.add_subplot(3, 2, 6)
            palette = np.zeros([201*201])
            for traj in expert_trajs:
                ax11.plot(traj.obs[:-1, 0])
                ax12.plot(traj.obs[:-1, 1])
                # ax11.set_ylim([-.4, .4])
                # ax12.set_ylim([-2.0, 2.0])
                # ax11.set_ylim([bound_dict["min_states"][0], bound_dict["max_states"][0]])
                # ax12.set_ylim([bound_dict["min_states"][1], bound_dict["max_states"][1]])
                ax21.plot(traj.obs[:-1, 0], traj.obs[:-1, 1])
                traj_idx = env.get_idx_from_obs(traj.obs[:-1])
                unique, counts = np.unique(traj_idx, return_counts=True)
                palette[unique] += counts
                # ax21.set_xlim([-.2, .20])
                # ax21.set_ylim([-1.20, 1.20])
                # ax21.plot(traj.obs[:-1, 2])
                # ax22.plot(traj.obs[:-1, 3])
                # ax21.set_ylim([-.3, .3])
                # ax22.set_ylim([-1., 1.])
                # ax21.set_ylim([bound_dict["min_states"][2], bound_dict["max_states"][2]])
                # ax22.set_ylim([bound_dict["min_states"][3], bound_dict["max_states"][3]])
                ax31.plot(traj.acts[:, 0])
                # ax32.plot(traj.acts[:, 1])
                # ax31.set_ylim([-10, 10])
                # ax32.set_ylim([-10, 10])
                # ax31.set_ylim([bound_dict["min_torques"][0], bound_dict["max_torques"][0]])
                # ax32.set_ylim([bound_dict["min_torques"][1], bound_dict["max_torques"][1]])
            # ax32.imshow(palette.reshape(201, 201))
        plt.tight_layout()
        plt.show()


def test_stepping_pkl_data():
    expert_dir = os.path.join("../../IRL", "demos", "DiscretizedPendulum", "301201_101_lqr", "quadcost_from_contlqr.pkl")
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
