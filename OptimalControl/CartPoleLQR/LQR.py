import numpy as np
import gym, math, torch, time
from matplotlib import pyplot as plt

def LQR(x0, u, n, model):
    K = BackwardPass(n, model)
    x, u, J = ForwardPass(x0, u, n, K, model)
    return x, u, J

def costfn(xt, ut, model):
    Q, R = model.Q, model.R
    cost = 0.5 * (xt.T @ Q @ xt + ut.T * R * ut)
    return cost.squeeze()

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
    model.reset()
    model.set_state(x0)
    model.render('human')
    for k in range(n):
        uk = u[k] + K[k] @ xk
        x_op[k], u_op[k] = xk, uk
        J[k] = costfn(xk, uk, model)
        ob, _, _, _ = model.step(uk[0])
        model.render('human')
        xk = ob.reshape(4, 1)
    model.close()
    return x_op, u_op, J

if __name__ == "__main__":
    model = gym.make("CartPoleCont-v0")
    n = 1000
    u = np.zeros([n, 1, 1])
    x0 = np.array([0, 0, 0, np.pi * 1/3]).reshape(4, 1)
    model.Q = np.diag([1, 1, 1, 1])
    model.R = 0.001
    x_op, u_op, J = LQR(x0, u, n, model)
    # plt.plot(u_op[:, 0])
    # plt.show()
    # plt.plot(x_op[:, 1])
    # plt.show()
    print("end")