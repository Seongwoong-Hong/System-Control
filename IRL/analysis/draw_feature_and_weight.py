import os
import torch as th
import numpy as np

from common.util import make_env, CPU_Unpickler

from matplotlib import pyplot as plt

irl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def draw_reward_weights():
    unnormalize = True
    log_dir = os.path.join(irl_path, "tmp", "log", "DiscretizedHuman", "MaxEntIRL", "sq_handnorm_")
    # get reward_weight and stack
    weights_stack = []
    label_name = [f"sub{i:02d}" for i in [6]]
    for subj in label_name:
        weights_per_actuation = []
        for actuation in range(1, 2):
            weights = []
            for trial in [1, 3, 4]:
                name = f"17171719_quadcost_finite_many/{subj}_{actuation}_{trial}/model"
                with open(log_dir + name + "/reward_net.pkl", "rb") as f:
                    reward_weight = CPU_Unpickler(f).load().layers[0].weight.detach()
                if unnormalize:
                    gains = feature_fn(th.Tensor([[0.16, 0.7, 0.8, 2.4, 1., 1.]]))
                    reward_weight /= gains
                weights.append(reward_weight.numpy().flatten())
            # weights = np.array(weights) / np.linalg.norm(np.array(weights), axis=-1, keepdims=True)  # normalize
            weights = np.array(weights)
            weights_per_actuation.append(
                np.append(weights.mean(axis=0, keepdims=True), weights.std(axis=0, keepdims=True), axis=0)
            )
        weights_stack.append(weights_per_actuation)
    weights_stack = np.array(weights_stack)
    w_mean = weights_stack[:, :, 0, :]
    w_std = weights_stack[:, :, 1, :]
    print(w_mean)

    # x1 = [f"{i}cm" for i in [3, 4.5, 6, 7.5]]#, 9, 12]]
    # subplot_name = [f"$\omega_{{{i + 1}}}$" for i in range(6)]
    subplot_name = [f"more expert"]
    x1 = [f"$\omega_{{{i + 1}}}$" for i in range(6)]
    # x = np.repeat([f"f{i}" for i in range(5, 9)], 5)
    for subj_idx, subj in enumerate(label_name):
        weight_fig = plt.figure(figsize=[6, 4], dpi=100.0)
        for weight_idx, weight_name in enumerate(subplot_name):
            ax = weight_fig.add_subplot(1, len(subplot_name), weight_idx + 1)
            # ax.errorbar(x1, w_mean[subj_idx, :, weight_idx], yerr=w_std[subj_idx, :, weight_idx], fmt='o')
            ax.errorbar(x1, w_mean[subj_idx, weight_idx, :], yerr=w_std[subj_idx, weight_idx, :], fmt='o')
            ax.legend([subj], ncol=1, columnspacing=0.1, fontsize=10)
            ax.set_xlabel("perturbation", labelpad=15.0, fontsize=14)
            ax.set_ylabel("weight", labelpad=15.0, fontsize=14)
            # ax.set_ylim(-0.3, .9)
            # ax.set_title(weight_name, fontsize=32)
            ax.tick_params(axis='both', which='major', labelsize=10)
            weight_fig.tight_layout()
        # plt.savefig(f"figures/disc_reward_weights/linspace_disc/{subj}_weights_notnorm.png")
        plt.show()


if __name__ == "__main__":
    def feature_fn(x):
        # return x
        return x ** 2
        # return th.cat([x, x ** 2], dim=1)

    # draw_feature_reward()
    draw_reward_weights()
