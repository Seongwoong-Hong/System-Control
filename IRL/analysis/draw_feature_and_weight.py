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
    # 그리고자하는 타입에 따라 수정해야 하는 변수들
    save_fig = False
    log_dir = os.path.join(irl_path, "tmp", "log")
    name = f"/HPC_custom/MaxEntIRL/sq_001alpha_stact"
    weights_indices = range(1, 7)
    actuations_indices = range(4, 6)
    subjects_indices = [5]
    trial_indices = range(1, 2)

    # weights = [rf"$\omega_{i}$" for i in range(1, 7)]
    weights = [r"$ank_{\theta}$", r"$hip_{\theta}$", r"$ank_{\dot{\theta}}$", r"$hip_{\dot{\theta}}$", r"$ank_{tq}$", r"$hip_{tq}$"]
    actuations = [f"$p_{i}$" for i in range(1, 8)]
    subjects = [f"sub{i:02d}" for i in range(1, 11)]

    weight_list = [weights[i - 1] for i in weights_indices]
    actu_list = [actuations[i - 1] for i in actuations_indices]
    subj_list = [subjects[i - 1] for i in subjects_indices]

    weights_stack = []
    for subj in subj_list:
        weights_per_actuation = []
        for actuation in actuations_indices:
            weights = []
            for trial in trial_indices:
                with open(log_dir + name + f"/{subj}_{actuation}_{trial}/model/reward_net.pkl", "rb") as f:
                    reward_weight = CPU_Unpickler(f).load().reward_layer.weight.detach()
                weights.append(reward_weight.square().detach().cpu().numpy().flatten())
            weights = np.array(weights)
            weights_per_actuation.append(
                np.append(weights.mean(axis=0, keepdims=True), weights.std(axis=0, keepdims=True), axis=0)
            )
        weights_stack.append(weights_per_actuation)

    # weight_stack은 [subject, actuation, mean or std, weight] 순서로 이루어짐
    weights_stack = np.array(weights_stack)
    # w_mean과 w_std는 trial에 대한 평균과 표준편차이며 순서는 [subject, actuation, weight]임
    w_mean = weights_stack[:, :, 0, :]
    w_std = weights_stack[:, :, 1, :]
    print(w_mean)

    fig = plt.figure(figsize=[15, 8], dpi=100.0)
    # 하나의 subplot에 들어가는 label로 구분되어 그려지는 데이터
    for label_idx, label_name in enumerate(subj_list):
        # 하나의 figure에 들어가는 subplot으로 구분되어 그려지는 데이터
        for subp_idx, subp_name in enumerate(weight_list):
            ax = fig.add_subplot(2, len(weight_list) // 2, 3 * (subp_idx % 2) + subp_idx // 2 + 1)
            # 하나의 subplot에 x vs y로 그려지는 데이터
            ax.errorbar(actu_list, w_mean[label_idx, :, subp_idx], yerr=w_std[label_idx, :, subp_idx], fmt='o-')
            # ax.set_xlabel("perturbation", labelpad=10.0, fontsize=14)
            # ax.set_ylabel("weight", labelpad=10.0, fontsize=14)
            ax.set_title(subp_name, fontsize=32)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 1))
            ax.tick_params(axis='both', which='both', labelsize=18)
    # figure를 보기 좋게 설정하는 옵션, 그리고자하는 figure에 따라 달라져야 함
    fig.axes[0].legend(subj_list, ncol=1, columnspacing=0.1)
    # weight_fig.axes[0].set_title(r'$\theta$', fontsize=44)
    # weight_fig.axes[2].set_title('$\omega$', fontsize=44)
    # weight_fig.axes[4].set_title('T', fontsize=44)
    # weight_fig.axes[0].set_ylabel("ank", fontsize=36)
    # weight_fig.axes[1].set_ylabel("hip", fontsize=36)
    # weight_fig.axes[1].tick_params(axis='x', which='major', bottom=True, labelbottom=True, labelsize=30)
    # weight_fig.axes[3].tick_params(axis='x', which='major', bottom=True, labelbottom=True, labelsize=30)
    # weight_fig.axes[5].tick_params(axis='x', which='major', bottom=True, labelbottom=True, labelsize=30)
    fig.tight_layout()

    if save_fig:
        figure_name = "figures" + name + "_weights.png"
        parent_dir = os.path.dirname(figure_name)
        os.makedirs(parent_dir, exist_ok=True)
        fig.savefig(figure_name)
    else:
        plt.show()


if __name__ == "__main__":
    def feature_fn(x):
        return x ** 2

    draw_reward_weights()
