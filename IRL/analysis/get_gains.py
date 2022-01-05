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
    weights_stack = []
    label_name = [f'sub{i:02d}' for i in [1, 3, 4, 5, 6, 7, 9, 10]]
    for subj in label_name:
        bsp = io.loadmat(irl_path + f"/demos/HPC/{subj}/{subj}i1.mat")['bsp']
        m2, l_u, h2, I2 = bsp[6, :]
        m_s, l_s, com_s, I_s = bsp[2, :]
        m_t, l_t, com_t, I_t = bsp[3, :]
        l_l = l_s + l_t
        m1 = 2 * (m_s + m_t)
        h1 = (m_s * com_s + m_t * (l_s + com_t)) / (m_s + m_t)
        I1 = 2 * (I_s + m_s * (h1 - com_s) ** 2 + I_t + m_t * (h1 - (l_s + com_t)) ** 2)
        A11 = I1 + m1 * h1 ** 2 + I2 + m2 * l_l ** 2 + 2 * m2 * l_l * h2 + m2 * h2 ** 2
        A12 = I2 + m2 * l_l * h2 + m2 * h2 ** 2
        A21 = A12
        A22 = I2 + m2 * h2 ** 2
        b1 = (m1 * h1 + m2 * l_l) * 9.81
        b2 = m2 * h2 * 9.81
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [(A22 * b1 + (A22 - A12) * b2) / (A11 * A22 - A21 * A12),
                       (A22 - A12) * b2 / (A11 * A22 - A21 * A12), 0, 0],
                      [(A21 * b1 + (A21 - A11) * b2) / (A12 * A21 - A11 * A22),
                       (A21 - A11) * b2 / (A12 * A21 - A11 * A22), 0, 0]])
        B = np.array([[0, 0],
                      [0, 0],
                      [A22 / (A11 * A22 - A21 * A12), -A12 / (A11 * A22 - A21 * A12)],
                      [A21 / (A12 * A21 - A11 * A22), -A12 / (A12 * A21 - A11 * A22)]])
        sub_stack = []
        for pert in range(1, 8):
            pert_stack = []
            for trial in range(1):
                name = f"/DiscretizedHuman_sq_09191927/{subj}_{pert}/model/000"
                with open(log_dir + name + "/reward_net.pkl", "rb") as f:
                    rwfn = pickle.load(f)
                weight = -rwfn.layers[0].weight.cpu().detach().numpy().flatten()
                # weight = -weight / np.linalg.norm(weight)
                Q = np.diag(weight[:4])
                Q[Q < 0] = 1e-4
                R = np.diag(weight[4:])
                R[R < 0] = 1e-6
                R[0, 0] *= (1 / 120) ** 2
                R[1, 1] *= (1 / 155) ** 2
                X = linalg.solve_continuous_are(A, B, Q, R)
                K = (np.linalg.inv(R) @ (B.T @ X))
                pert_stack.append(K.flatten())
            sub_stack.append(np.array(pert_stack))
            # mean_stack = np.array(pert_stack).mean(axis=0, keepdims=True)
            # std_stack = np.array(pert_stack).std(axis=0, keepdims=True)
            # sub_stack.append(np.append(mean_stack, std_stack, axis=0))
        weights_stack.append(sub_stack)
    weights_stack = np.array(weights_stack)
    w_mean = weights_stack.mean(axis=0)
    w_std = weights_stack.std(axis=0)
    subplot_name = [
        r"$T_{ank}/\theta_{ank}$", r"$T_{ank}/\theta_{hip}$", r"$T_{ank}/\dot\theta_{ank}$",
        r"$T_{ank}/\dot\theta_{hip}$",
        r"$T_{hip}/\theta_{ank}$", r"$T_{hip}/\theta_{hip}$", r"$T_{hip}/\dot\theta_{ank}$",
        r"$T_{hip}/\dot\theta_{ank}$"
    ]
    x = [i for i in ['3', '4.5', '6', '7.5', '9', '12', '15']]
    # x = np.repeat([f"f{i}" for i in range(5, 9)], 5)
    fig = plt.figure(figsize=[18, 15], dpi=150.0)
    for i in range(len(subplot_name)):
        ax = fig.add_subplot(2, 4, i + 1)
        # for subj in range(len(label_name)):
        # ax.scatter(x, weigths_stack[i, j*5:(j+1)*5, 4])
        # ax.scatter(x, weigths_stack[j, :, i])
        ax.errorbar(x, w_mean[:, 0, i], yerr=w_std[:, 0, i], fmt='o')
        # ax.scatter(x, weigths_stack[1, :, i])
        # ax.legend(label_name, ncol=1, columnspacing=0.1, fontsize=15)
        ax.set_xlabel("Perturbation(cm)", labelpad=15.0, fontsize=24)
        ax.set_ylabel("gain", labelpad=15.0, fontsize=24)
        ax.set_title(subplot_name[i], fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=15)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    def feature_fn(x):
        # return x
        return x ** 2
        # return th.cat([x, x ** 2], dim=1)
        # return th.cat([x, x**2, x**3, x**4], dim=1)


    # draw_feature_reward()
    get_gains()
