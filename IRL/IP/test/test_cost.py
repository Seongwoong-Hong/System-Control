import os, torch, gym, gym_envs
import numpy as np
from algo.torch.OptCont import LQRPolicy
from common.modules import NNCost
from matplotlib import pyplot as plt

class ExpertPolicy(LQRPolicy):
    def _build_env(self):
        m, g, h, I = 5.0, 9.81, 0.5, 1.667
        self.Q = np.array([[1, 0], [0, 1]])
        self.R = 0.001
        self.A = np.array([[0, m*g*h/I], [1, 0]])
        self.B = np.array([[1/I], [0]])
        return self.A, self.B, self.Q, self.R

model_dir = os.path.join("..", "tmp", "model")
costfn = torch.load(model_dir + "/bar7_costfn.pt")
env = gym.make("IP_custom-v2", n_steps=100)
exp = ExpertPolicy(env)
dt = env.dt

th, dth = np.meshgrid(np.linspace(-0.2, 0.2, 100), np.linspace(-0.2, 0.2, 100))
act = -exp.P*th - exp.D*dth
cost_agt = np.zeros(th.shape)
cost_exp = np.zeros(th.shape)

for i in range(th.shape[0]):
    for j in range(th.shape[1]):
        cost_exp[i][j] = np.array([th[i][j], dth[i][j]])@exp.Q@np.array([th[i][j], dth[i][j]]).T+act[i][j]*act[i][j]*exp.R
        inp = torch.Tensor([th[i][j], dth[i][j], act[i][j]]).double()
        cost_agt[i][j] =  costfn(inp).item()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(th, dth, cost_agt)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(th, dth, cost_exp)
plt.show()