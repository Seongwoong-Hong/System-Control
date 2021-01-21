import os, torch, gym, gym_envs
import numpy as np
from algo.torch.OptCont import LQRPolicy
from algo.torch.ppo import PPO
from common.modules import NNCost
from matplotlib import pyplot as plt
from matplotlib import cm

class ExpertPolicy(LQRPolicy):
    def _build_env(self):
        m, g, h, I = 5.0, 9.81, 0.5, 1.667
        self.Q = np.array([[1, 0], [0, 1]])
        self.R = 0.001
        self.A = np.array([[0, m*g*h/I], [1, 0]])
        self.B = np.array([[1/I], [0]])
        return self.A, self.B, self.Q, self.R

name = "bar3"
num = 50
model_dir = os.path.join("..", "tmp", "model", name)
costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
env = gym.make("IP_custom-v1", n_steps=100)
exp = ExpertPolicy(env)

dt = env.dt

th, dth = np.meshgrid(np.linspace(-0.5, 0.5, 100), np.linspace(-0.5, 0.5, 100))
pact = np.zeros((100, 100), dtype=np.float64)
act = -exp.P*th - exp.D*dth
cost_agt = np.zeros(th.shape)
cost_exp = np.zeros(th.shape)

for i in range(th.shape[0]):
    for j in range(th.shape[1]):
        pact[i][j], _ = algo.predict(np.append(th[i][j], dth[i][j]), deterministic=True)
        cost_exp[i][j] = np.array([th[i][j], dth[i][j]])@exp.Q@np.array([th[i][j], dth[i][j]]).T+act[i][j]*act[i][j]*exp.R
        inp = torch.Tensor([th[i][j], dth[i][j], act[i][j]]).double()
        cost_agt[i][j] = costfn(inp).item()

cost_agt /= np.amax(cost_agt)
cost_exp /= np.amax(cost_exp)
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
surf1 = ax1.plot_surface(th, dth, cost_agt, cmap=cm.coolwarm)
surf2 = ax2.plot_surface(th, dth, cost_exp, cmap=cm.coolwarm)
surf3 = ax3.plot_surface(th, dth, pact, cmap=cm.coolwarm)
surf4 = ax4.plot_surface(th, dth, act, cmap=cm.coolwarm)
fig.colorbar(surf1, ax=ax1)
fig.colorbar(surf2, ax=ax2)
fig.colorbar(surf3, ax=ax3)
fig.colorbar(surf4, ax=ax4)
ax1.view_init(azim=0, elev=90)
ax2.view_init(azim=0, elev=90)
ax3.view_init(azim=0, elev=90)
ax4.view_init(azim=0, elev=90)
plt.show()