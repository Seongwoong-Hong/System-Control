import numpy as np
import gym, math, torch, time
from matplotlib import pyplot as plt

def iLQR(x, u, model):
    n = u[:, 0, 0].__len__()
    J = np.zeros([n, 1])
    for idx, value in enumerate(np.concatenate((x, u), 1)):
        J[idx] = costfn(value[:-1], value[-1], model)
    while(True):
        J_prev = J
        K, d = BackwardPass(x, u, model)
        x, u, J = ForwardPass(x, u, K, d, model)
        print("Current and Previous Cost difference is %.2f" %(np.linalg.norm(J - J_prev)))
        if (np.linalg.norm(J - J_prev) < 0.01):
            break
    return x, u, J

def costfn(xt, ut, model):
    Q, R = model.Q, model.R
    cost = 0.5 * (xt.T @ Q @ xt + ut.T * R * ut)
    return cost.squeeze()

def BackwardPass(x, u, model):
    n = u[:, 0, 0].__len__()
    ns = x[0, :, 0].__len__()
    na = u[0, :, 0].__len__()
    Q, R = model.Q, model.R
    Vk, vk = Q, Q @ x[-1]
    K, d = np.zeros([n, 1, ns]), np.zeros([n, na])
    for k in range(n):
        idx = n - 1 - k
        Fk = cal_dynamics(np.concatenate((x[idx], u[idx]), 0), model)
        Qk = np.block([[Q, np.zeros([4, 1])],[np.zeros([1, 4]), R]]) + Fk.T @ Vk @ Fk
        qk = np.block([[Q, np.zeros([4, 1])],[np.zeros([1, 4]), R]]) @ np.concatenate((x[idx], u[idx]), 0) + Fk.T @ vk
        # start = time.time()
        K[idx] = -(Qk[-1, -1] ** -1) * Qk[-1, 0:-1]
        d[idx] = -(Qk[-1, -1] ** -1) * qk[-1]
        Vk = Qk[0:-1, 0:-1] + Qk[0:-1, [-1]] @ K[idx] + K[idx].T @ Qk[[-1], 0:-1] + K[idx].T * Qk[-1, -1] @ K[idx]
        vk = qk[0:-1] + Qk[0:-1, [-1]] * d[idx] + K[idx].T * qk[-1] + K[idx].T * Qk[-1, -1] * d[idx]
        # print(time.time()-start)
    return K, d

def ForwardPass(x, u, K, d, model):
    n = u.__len__()
    J = np.zeros(n)
    x_op = np.zeros(x.shape)
    u_op = np.zeros(u.shape)
    xk = x[0]
    model.set_state(xk)
    for k in range(n):
        uk = u[k] + (K[k] @ (xk - x[k]) + d[k])
        x_op[k], u_op[k] = xk, uk
        J[k] = costfn(xk, uk, model)
        ob, _, _, _ = model.step(uk[0])
        xk = ob.reshape(4, 1)
    return x_op, u_op, J

def cal_dynamics(state, model):
    state = torch.tensor(state, dtype=torch.float, requires_grad=True)
    mc, mp, g, l, h = model.masscart, model.masspole, model.gravity, model.length, model.tau
    Fk = np.zeros([4, 5])
    for i in range(4):
        dx, x, dth, th, u = state
        sinth = math.sin(th)
        costh = math.cos(th)
        temp = (u + mp * l * dth ** 2 * sinth) / (mp + mc)
        ddth = (g * sinth - costh * temp) / (l * (4.0 / 3.0 - mp * costh ** 2 / (mp + mc)))
        ddx = temp - mp * l * ddth * costh / (mp + mc)
        x_new = x + dx * h
        dx_new = dx + ddx * h
        th_new = th + dth * h
        dth_new = dth + ddth * h
        new_state = torch.cat((dx_new, x_new, dth_new, th_new))
        new_var = new_state[i]
        new_var.backward()
        Fk[i] = state.grad.data.numpy().squeeze()
        state.grad.data.zero_()
    return Fk

if __name__ == "__main__":
    model = gym.make("CartPoleCont-v0")
    n, ns, na = 1000, 4, 1
    x, u = np.zeros([n, ns, 1]), np.zeros([n, na, 1])
    x[0] = model.reset().reshape(ns, 1)
    x0 = np.array([0, 0, 0, np.pi * 1/12]).reshape(4, 1)
    model.Q = np.diag([1, 1, 1, 1])
    model.R = 0.001
    model.set_state(x0)
    model.render('human')
    for i in range(n - 1):
        ob, _, _, _ = model.step(u[i][0])
        model.render('human')
        x[i + 1] = ob.reshape(ns, 1)
    model.close()

    x_op, u_op, J = iLQR(x, u, model)
    x[0] = x_op[0]
    model.set_state(x[0])
    # for i in range(n - 1):
    #     model.render('human')
    #     model.step(u_op[i])

    plt.plot(x_op[:,1])
    plt.show()