import numpy as np
import gym, math
from matplotlib import pyplot as plt

def iLQR(x, u):
    n = u.__len__()
    J = np.zeros([n, 1])
    x = x.reshape(n, 2, 1)
    u = u.reshape(n, 1, 1)
    for idx, value in enumerate(np.concatenate((x, u), 1)):
        J[idx] = costfn(value[:-1], value[-1])
    while(True):
        J_prev = J
        K, Kv, Ku, S, v = BackwardPass(x, u)
        x, u, J = ForwardPass(x, u, K, Kv, Ku, S, v)
        if (np.linalg.norm(J - J_prev) < 1):
            break
    return x, u, J

def costfn(xt, ut):
    Q = np.array([[1, 0], [0, 1]])
    R = 0.1
    cost = 0.5 * (xt.T @ Q @ xt + ut.T * R * ut)
    return cost.squeeze()


def BackwardPass(x, u):
    n = u.__len__()
    Q = np.array([[1, 0], [0, 1]])
    R = 0.1
    K, Kv, Ku = np.zeros([n, 1, 2]), np.zeros([n, 1, 2]), np.zeros(n)
    S, v = np.zeros([n, 2, 2]), np.zeros([n, 2, 1])
    Sk, vk = Q, Q @ x[-1]
    for k in range(n):
        idx = n - 1 - k
        Ak, Bk = cal_dynamics(x[idx], u[idx][0])
        Kk = np.linalg.inv(Bk.T @ Sk @ Bk + R) @ Bk.T @ Sk @ Ak
        Kvk = np.linalg.inv(Bk.T @ Sk @ Bk +R) @ Bk.T
        Kuk = np.linalg.inv(Bk.T @ Sk @ Bk + R) * R
        K[idx] = Kk
        Kv[idx]= Kvk
        Ku[idx] = Kuk
        S[idx] = Sk
        v[idx] = vk
        Sk = Ak.T @ Sk @ (Ak - Bk @ Kk) + Q
        vk = (Ak - Bk @ Kk).T @ vk - Kk.T * R * u[idx][0] + Q @ x[idx]

    return K, Kv, Ku, S, v

def ForwardPass(x, u, K, Kv, Ku, S, v):
    n = u.__len__()
    J = np.zeros(n)
    x_op = np.zeros([n, 2, 1])
    u_op = np.zeros([n, 1, 1])
    xk = x[0]
    model = gym.make("IP_custom-v2")
    model.set_state(xk[1], xk[0])
    for k in range(n):
        uk = u[k][0] -K[k] @ (xk - x[k]) - Kv[k] @ v[k] - Ku[k] * u[k][0]
        x_op[k], u_op[k] = xk, uk
        J[k] = costfn(xk, uk)
        ob, _, _, _ = model.step(uk)
        xk = ob.reshape(2, 1)
    return x_op, u_op, J

def cal_dynamics(x, u):
    dxk = x[0]
    xk = x[1]
    m, g, h, I = 5, 9.81, 0.8, 4.27
    dT = 1e-3
    pert = 1e-6
    dxk_n = dxk + dT * ((m * g * h) / I * math.sin(xk) + u / I)
    xk_n = xk + dT * dxk
    dxk_vpert = dxk + pert + dT * ((m * g * h) / I * math.sin(xk) + u / I)
    xk_vpert = xk + dT * (dxk + pert)
    dxk_xpert = dxk + dT * ((m * g * h) / I * math.sin(xk + pert) + u / I)
    xk_xpert = xk + pert + dT * dxk
    dxk_upert = dxk + dT * ((m * g * h) / I * math.sin(xk) + (u + pert) / I)
    xk_upert = xk + dT * dxk
    Ak = np.array([[(dxk_vpert - dxk_n) / pert, (dxk_xpert - dxk_n) / pert], [(xk_vpert - xk_n) / pert, (xk_xpert - xk_n) / pert]])
    Bk = np.array([[(dxk_upert - dxk_n) / pert], [(xk_upert - xk_n) / pert]])
    return Ak.squeeze(-1), Bk.squeeze(-1)

if __name__ == "__main__":
    model = gym.make("IP_custom-v2")
    n = 3000
    x, u = np.zeros([n, 2, 1]), np.zeros([n, 1, 1])
    x[0] = model.reset().reshape(2, 1)
    for i in range(2999):
        ob, _, _, _ = model.step(u[i][0])
        x[i + 1] = ob.reshape(2, 1)

    x_op, u_op, J = iLQR(x, u)
    plt.plot(x_op[:, 1])
    plt.show()