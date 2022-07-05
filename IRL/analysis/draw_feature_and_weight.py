import os
import json
import torch as th
import numpy as np

from common.util import make_env, CPU_Unpickler

from matplotlib import pyplot as plt

irl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
with open(f"{irl_path}/demos/bound_info.json", "r") as f:
    bound_info = json.load(f)


def draw_reward_weights():
    unnormalize = True
    log_dir = os.path.join(irl_path, "tmp", "log", "DiscretizedHuman", "MaxEntIRL", "ext_01alpha_")
    # get reward_weight and stack
    weights_stack = []
    label_name = [f"sub{i:02d}" for i in [5]]
    for subj in label_name:
        weights_per_actuation = []
        for actuation in [4]:
            weights = []
            for trial in range(1, 5):#
                name = f"19191919_quadcost_disc/{subj}_{actuation}_0001alpha_{trial}/model"
                # name = f"50/1alpha_{trial}/model"
                with open(log_dir + name + "/reward_net.pkl", "rb") as f:
                    reward_weight = CPU_Unpickler(f).load().layers[0].weight.detach()
                if unnormalize:
                    perturbation = actuation - 1
                    bound_dict = bound_info[subj][perturbation]
                    normalizer = np.max(np.append(np.append(bound_dict["max_states"], bound_dict["max_torques"])[None, :],
                                                  abs(np.append(bound_dict["min_states"], bound_dict["min_torques"]))[None, :],
                                                  axis=0),
                                        axis=0, keepdims=True)
                    normalizer[0, -2:] = 1
                    # normalizer = np.array([[49, 49, 6, 6]])
                    gains = feature_fn(th.from_numpy(normalizer).float())
                    # reward_weight /= gains
                weights.append(reward_weight.numpy().flatten())
            weights = np.array(weights)
            # weights = np.array(weights) / np.linalg.norm(np.array(weights), axis=-1, keepdims=True)  # normalize
            weights_per_actuation.append(
                np.append(weights.mean(axis=0, keepdims=True), weights.std(axis=0, keepdims=True), axis=0)
            )
        weights_stack.append(weights_per_actuation)
    weights_stack = np.array(weights_stack)
    w_mean = weights_stack[:, :, 0, :]
    w_std = weights_stack[:, :, 1, :]
    print(w_mean)

    # x1 = [f"{i}cm" for i in [3, 4.5, 6, 7.5]]#, 9, 12]]
    # subplot_name = [f"$\omega_{{{i + 1}}}$" for i in range(12)]
    # subplot_name = [f"more expert"]
    subplot_name = [f"{i}" for i in [1]]
    x1 = [f"$\omega_{{{i + 1}}}$" for i in range(15)]
    # x1 = [f"{i}" for i in range(1, 6)]
    # x1 = [r"$\theta_1$", r"$\theta_2$", r"$\dot{\theta_1}$", r"$\dot{\theta_2}$", "$T_1$", "$T_2$"]
    # x = np.repeat([f"f{i}" for i in range(5, 9)], 5)
    for subj_idx, subj in enumerate(label_name):
        weight_fig = plt.figure(figsize=[5, 4], dpi=100.0)
        for weight_idx, weight_name in enumerate(subplot_name):
            ax = weight_fig.add_subplot(1, len(subplot_name), weight_idx + 1)
            # ax.errorbar(x1, w_mean[subj_idx, :, weight_idx], yerr=w_std[subj_idx, :, weight_idx], fmt='o')
            ax.errorbar(x1, w_mean[subj_idx, weight_idx, :], yerr=w_std[subj_idx, weight_idx, :], fmt='o')
            # ax.legend([subj], ncol=1, columnspacing=0.1, f=1)
            ax.set_xlabel("states", labelpad=10.0, fontsize=14)
            ax.set_ylabel("weight", labelpad=10.0, fontsize=14)
            # ax.set_ylim(-2.6, 0.1)
            # ax.set_title(weight_name, fontsize=32)
            plt.tick_params(axis='x', which='major', labelsize=14)
            plt.tick_params(axis='y', which='major', labelsize=10)
        weight_fig.tight_layout()
        # plt.savefig(f"figures/disc_reward_weights/linspace_disc/{subj}_weights_notnorm.png")
        plt.show()


if __name__ == "__main__":
    def feature_fn(x):
        # return x
        # return x ** 2
        return th.cat([x, x ** 2], dim=1)
        # x1, x2, x3, x4, a1, a2 = th.split(x, 1, dim=1)
        # return th.cat((x, x ** 2, x1 * x2, x3 * x4, a1 * a2), dim=1)

    draw_reward_weights()
