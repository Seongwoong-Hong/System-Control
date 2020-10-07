import numpy as np
import gym, math
from matplotlib import pyplot as plt

def LQR(x0, u, n):
    K = BackwardPass(n)
    x, u, J = ForwardPass(x0, u, n, K)
    return x, u, J

def costfn(xt, ut):
    Q = np.array([[1, 0], [0, 1]])
    R = 0.001
    cost = 0.5 * (xt.T @ Q @ xt + ut.T * R * ut)
    return cost.squeeze()

def BackwardPass(n):
    Q = np.array([[1, 0], [0, 1]])
    R = 0.001
    m, g, h, I, dT = 5, 9.81, 0.8, 4.27, 1e-3
    Vk = Q
    K = np.zeros([n, 1, 2])
    Ak, Bk = np.array([[1, dT * m * g * h / I], [dT, 1]]), np.array([[dT / I], [0]])
    Fk = np.concatenate((Ak, Bk), 1)
    for k in range(n):
        idx = n - 1 - k
        Qk = np.diag([1, 1, 0.01]) + Fk.T @ Vk @ Fk
        K[idx] = -(Qk[-1, -1] ** -1) * Qk[-1, 0:-1]
        Vk = Qk[0:2, 0:2] + Qk[0:2, [-1]] @ K[idx] + K[idx].T @ Qk[[-1], 0:2] + K[idx].T * Qk[-1, -1] @ K[idx]
    return K

def ForwardPass(x0, u, n, K):
    J = np.zeros(n)
    x_op = np.zeros([n, 2, 1])
    u_op = np.zeros([n, 1, 1])
    xk = x0
    model = gym.make("IP_custom-v2")
    model.set_state(xk[1], xk[0])
    for k in range(n):
        uk = u[k] + K[k] @ xk
        x_op[k], u_op[k] = xk, uk
        J[k] = costfn(xk, uk)
        ob, _, _, _ = model.step(uk)
        xk = ob.reshape(2, 1)
    return x_op, u_op, J

if __name__ == "__main__":
    model = gym.make("IP_custom-v2")
    n = 6000
    u = np.zeros([n, 1, 1])
    x0 = model.reset().reshape(2, 1)
    x_op, u_op, J = LQR(x0, u, n)
    plt.plot(x_op[:, 1])
    plt.show()
    plt.plot(J)
    plt.show()