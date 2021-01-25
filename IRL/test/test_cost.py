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

name = "2021-1-24-18-46-3"
nums = np.linspace(2, 20, 10)
for num in nums:
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    costfn = torch.load(model_dir + "/costfn%d.pt"%(num)).to('cpu')
    algo = PPO.load(model_dir + "/ppo%d.zip"%(num))
    # algo = PPO.load(model_dir + "/extra_ppo.zip".format(name))
    env = gym.make("IP_custom-v1", n_steps=100)
    exp = ExpertPolicy(env)
    dt = env.dt
    size = 100
    th, dth = np.meshgrid(np.linspace(-0.3, 0.3, size), np.linspace(-0.3, 0.3, size))
    pact = np.zeros((size, size), dtype=np.float64)
    act = np.zeros((size, size), dtype=np.float64)
    cost_agt = np.zeros(th.shape)
    cost_exp = np.zeros(th.shape)

    for i in range(th.shape[0]):
        for j in range(th.shape[1]):
            pact[i][j], _ = algo.predict(np.append(th[i][j], dth[i][j]), deterministic=True)
            act[i][j], _ = exp.predict(np.append(th[i][j], dth[i][j]), deterministic=True)
            cost_exp[i][j] = np.array([th[i][j], dth[i][j]])@exp.Q@np.array([th[i][j], dth[i][j]]).T+act[i][j]*act[i][j]*exp.R*exp.gear**2
            inp = torch.Tensor((th[i][j], dth[i][j], act[i][j])).double()
            cost_agt[i][j] = costfn(inp).item()

    cost_agt /= np.amax(cost_agt)
    cost_exp /= np.amax(cost_exp)
    title_list = ["norm_cost", "norm_cost", "abs_action", "abs_action"]
    yval_list = [cost_agt, cost_exp, np.abs(pact), np.abs(act)]
    xlabel, ylabel = "theta(rad)", "dtheta(rad/s)"
    max_list = [1.0, 1.0, 0.3, 0.3]
    min_list = [0.0, 0.0, 0.0, 0.0]
    fig = plt.figure()
    for i in range(4):
        ax = fig.add_subplot(2, 2, (i+1))
        surf = ax.pcolor(th, dth, yval_list[i], cmap=cm.coolwarm, shading='auto', vmax=max_list[i], vmin=min_list[i])
        clb = fig.colorbar(surf, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        clb.ax.set_title(title_list[i])
    plt.show()