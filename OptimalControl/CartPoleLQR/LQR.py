import numpy as np
import gym, time, gym_envs
from matplotlib import pyplot as plt


def LQR(x0, u, n, model):
    K = BackwardPass(n, model)
    x, u, J = ForwardPass(x0, u, n, K, model)
    return x, u, J


def BackwardPass(n, model):
    Q, R = model.Q, model.R
    mc, mp, g, l, h = model.masscart, model.masspole, model.gravity, model.length, model.tau
    Vk = Q
    K = np.zeros([n, 1, 4])
    Ak, Bk = np.array([[1, 0, 0, -h * mp*l/(mc + mp) * g/(l * (4/3 - mp/(mc + mp)))], [h, 1, 0, 0], [0, 0, 1, h * g/(l * (4/3 - mp/(mc + mp)))], [0, 0, h, 1]]), np.array([[h/(mc + mp)], [0], [-h/(l * 4/3 * mc + 1/3 * mp)], [0]])
    Fk = np.concatenate((Ak, Bk), 1)
    for k in range(n):
        idx = n - 1 - k
        Qk = np.block([[Q, np.zeros([4, 1])],[np.zeros([1, 4]), R]]) + Fk.T @ Vk @ Fk
        K[idx] = -(Qk[-1, -1] ** -1) * Qk[-1, 0:-1]
        Vk = Qk[0:-1, 0:-1] + Qk[0:-1, [-1]] @ K[idx] + K[idx].T @ Qk[[-1], 0:-1] + K[idx].T * Qk[-1, -1] @ K[idx]
    return K


def ForwardPass(x0, u, n, K, model):
    J = np.zeros(n)
    x_op = np.zeros([n, 4, 1])
    u_op = np.zeros([n, 1, 1])
    xk = x0
    frames = []
    model.reset()
    model.set_state(x0)
    frames.append(model.render('rgb_array'))
    for k in range(n):
        uk = u[k] + K[k] @ xk
        x_op[k], u_op[k] = xk, uk
        ob, cost, _, _ = model.step(uk[0])
        frames.append(model.render('rgb_array'))
        xk = ob.reshape(4, 1)
        J[k] = cost
    model.saving_gif(name="LQR.gif", frames=frames)
    model.close()
    return x_op, u_op, J


if __name__ == "__main__":
    n = 1000
    model = gym.make(id="CartPoleContTest-v0", max_ep=n)
    u = np.zeros([n, 1, 1])
    x0 = np.array([0, 0, 0, np.pi * 1/4]).reshape(4, 1)
    model.Q = np.diag([1, 1, 1, 1])
    model.R = 0.001
    x_op, u_op, J = LQR(x0, u, n, model)
    plt.plot(J)
    plt.show()
    plt.plot(u_op.squeeze())
    plt.show()
    print(np.sum(J))