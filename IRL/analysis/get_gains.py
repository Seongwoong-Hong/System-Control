import os
import pickle
import numpy as np
import torch as th
from scipy import io, linalg
from matplotlib import pyplot as plt

irl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_gains():
    log_dir = os.path.join(irl_path, "tmp", "log", "ray_result")
    # get reward_weight and stack
    weigths_stack = []
    for subj in ["sub03", "sub07"]:
        bsp = io.loadmat(irl_path + f"/demos/HPC/{subj}/{subj}i1.mat")['bsp']
        m2, l_u, h2, I2 = bsp[6, :]
        m_s, l_s, com_s, I_s = bsp[2, :]
        m_t, l_t, com_t, I_t = bsp[3, :]
        l_l = l_s + l_t
        m1 = 2 * (m_s + m_t)
        h1 = (m_s * com_s + m_t * (l_s + com_t)) / (m_s + m_t)
        I1 = 2 * (I_s + m_s * (h1 - com_s) ** 2 + I_t + m_t * (h1 - (l_s + com_t)) ** 2)
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [m1 * 9.81 * h1 / I1, 0, 0, 0],
                      [0, m2 * 9.81 * h2 / I2, 0, 0]])
        B = np.array([[0, 0],
                      [0, 0],
                      [1 / I1, -1 / I1],
                      [0, 1 / I2]])
        sub_stack = []
        for pert in [1, 2, 3]:
            for trial in [1, 2, 3, 4, 5]:
                name = f"/DiscretizedHuman_{subj}/{subj}_{pert}_{trial}/model/000"
                with open(log_dir + name + "/reward_net.pkl", "rb") as f:
                    rwfn = pickle.load(f)
                Q = np.diag(rwfn.layers[0].weight.detach().numpy().flatten()[4:])
                R = np.diag([2e-2, 2e-2])
                X = linalg.solve_continuous_are(A, B, Q, R)
                K = (np.linalg.inv(R) @ (B.T @ X))
                sub_stack.append(K.flatten())
        weigths_stack.append(sub_stack)
    weigths_stack = np.array(weigths_stack)
    subplot_name = [
        r"$T_{ank}/\theta_{ank}$", r"$T_{ank}/\theta_{hip}$", r"$T_{ank}/\dot\theta_{ank}$",
        r"$T_{ank}/\dot\theta_{hip}$",
        r"$T_{hip}/\theta_{ank}$", r"$T_{hip}/\theta_{hip}$", r"$T_{hip}/\dot\theta_{ank}$",
        r"$T_{hip}/\dot\theta_{ank}$"
    ]
    x = [f"Pert_{i // 5 + 1}" for i in range(15)]
    # x = np.repeat([f"f{i}" for i in range(5, 9)], 5)
    fig = plt.figure(figsize=[36, 12], dpi=300.0)
    for i in range(len(subplot_name)):
        ax = fig.add_subplot(2, 4, i + 1)
        for j in range(1):
            # ax.scatter(x, weigths_stack[i, j*5:(j+1)*5, 4])
            ax.scatter(x, weigths_stack[0, :, i])
            ax.scatter(x, weigths_stack[1, :, i])
        ax.legend(["sub01", "sub02"], ncol=1, columnspacing=0.1, fontsize=15)
        ax.set_xlabel("Perturbation", labelpad=15.0, fontsize=24)
        ax.set_ylabel("weight", labelpad=15.0, fontsize=24)
        ax.set_title(subplot_name[i], fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=15)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    def feature_fn(x):
        # return x
        return th.cat([x, x ** 2], dim=1)
        # return th.cat([x, x**2, x**3, x**4], dim=1)


    # draw_feature_reward()
    get_gains()
