import os
import numpy as np
from scipy import io, signal, linalg
from matplotlib import pyplot as plt

from common.sb3.util import make_env

irl_path = os.path.abspath("../../IRL")
basis_data = os.path.join(irl_path, "demos", "HPC", "sub01_full", "sub01i16.mat")
bsp = io.loadmat(basis_data)['bsp']
pltq = io.loadmat(basis_data)['pltq']
init_state = io.loadmat(basis_data)['state'][0, :4]
env = make_env("HPC_custom-v2", bsp=bsp, init_states=[init_state], pltqs=[pltq / 300])

I1, I2 = env.model.body_inertia[1:, 0]
l1 = env.model.body_pos[2, 2]
lc1, lc2 = env.model.body_ipos[1:, 2]
m1, m2 = env.model.body_mass[1:]
g = 9.81
M = np.array([[I1 + m1 * lc1 ** 2 + I2 + m2 * l1 ** 2 + 2 * m2 * l1 * lc2 + m2 * lc2 ** 2,
               I2 + m2 * l1 * lc2 + m2 * lc2 ** 2],
              [I2 + m2 * l1 * lc2 + m2 * lc2 ** 2, I2 + m2 * lc2 ** 2]])
C = np.array([[m1 * lc1 * g + m2 * l1 * g + m2 * g * lc2, m2 * g * lc2],
              [m2 * g * lc2, m2 * g * lc2]])
Ac, Bc = np.zeros([4, 4]), np.zeros([4, 2])
Ac[:2, 2:] = np.eye(2, 2)
Ac[2:, :2] = np.linalg.inv(M) @ C
Bc[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
Ad, Bd, _, _, dt = signal.cont2discrete((Ac, Bc, np.array([1, 1, 1, 1]), 0), env.dt)

Q = np.diag([0.6783, 0.2000, 0.1000, 0.05])
R = np.diag([0.0001, 0.0001])


def test_compare_lsim_dlsim():
    Xc = linalg.solve_continuous_are(Ac, Bc, Q, R)
    Xd = linalg.solve_discrete_are(Ad, Bd, Q, R)
    Kc = np.linalg.inv(R) @ (Bc.T @ Xc)
    Kd = np.linalg.inv(Bd.T @ Xd @ Bd + R) @ (Bd.T @ Xd)
    t, _, xc = signal.lsim((Ac - Bc @ Kc, Bc, np.eye(4, 4), np.zeros([4, 2])), pltq, np.arange(0, 3, 1/120), init_state)
    t, _, xd = signal.dlsim((Ad - Bd @ Kd, Bd, np.eye(4, 4), np.zeros([4, 2]), 1/120), pltq, np.arange(0, 3, 1/120), init_state)
    uc = xc @ -Kc.T
    ud = xd @ -Kd.T
    fig = plt.figure()
    for i in range(4):
        ax = fig.add_subplot(3, 2, i+1)
        ax.plot(t, xc[:, i])
        ax.plot(t, xd[:, i])
    for i in range(2):
        ax = fig.add_subplot(3, 2, i+5)
        ax.plot(t, uc[:, i])
        ax.plot(t, ud[:, i])
    plt.show()


def test_compare_lsim_step():
    Xd = linalg.solve_discrete_are(Ad, Bd, Q, R)
    Kd = np.linalg.inv(Bd.T @ Xd @ Bd + R) @ (Bd.T @ Xd)
    t, _, xd = signal.dlsim((Ad - Bd @ Kd, Bd, np.eye(4, 4), np.zeros([4, 2]), 1/120), pltq, np.arange(0, 3, 1/120), init_state)
    ud = xd @ -Kd.T
    ob = env.reset()
    done = False
    obs, acts = [ob], []
    while not done:
        a = (ob @ -Kd.T)
        ob, _, done, _ = env.step(a / 300)
        obs.append(ob)
        acts.append(a)
    obs = np.array(obs)
    acts = np.array(acts)
    fig1 = plt.figure()
    for i in range(4):
        ax = fig1.add_subplot(4, 2, i+1)
        ax.plot(t, xd[:, i])
        ax.plot(t, obs[:-1, i])
    for i in range(2):
        ax = fig1.add_subplot(4, 2, i+5)
        ax.plot(t, ud[:, i])
        ax.plot(t, acts[:, i])
    ax = fig1.add_subplot(4, 2, 7)
    ax.plot(t, pltq[:, 0])
    ax = fig1.add_subplot(4, 2, 8)
    ax.plot(t, pltq[:, 1])
    plt.show()
