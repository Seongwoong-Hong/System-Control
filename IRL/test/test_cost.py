import gym_envs
import os
import torch

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from IRL.project_policies import def_policy
from algo.torch.ppo import PPO


env_type = "IDP"
name = "{}/2021-2-5-21-4-22".format(env_type)
nums = np.linspace(2, 10, 5)
# nums = [2]
for num in nums:
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    costfn = torch.load(model_dir + "/costfn%d.pt" % num).to('cpu')
    algo = PPO.load(model_dir + "/ppo%d.zip" % num)
    # algo = PPO.load(model_dir + "/extra_ppo.zip".format(name))
    env = gym_envs.make("{}_custom-v1".format(env_type), n_steps=200)
    draw_dim = [0, 1, 0]
    dim_names = ["q1(rad)", "q2(rad)", "dq1(rad/s)", "dq2(rad/s)"]
    exp = def_policy(env_type, env)
    dt = env.dt
    size = 100

    d1, d2 = np.meshgrid(np.linspace(-0.3, 0.3, size), np.linspace(-0.3, 0.3, size))
    pact = np.zeros((size, size), dtype=np.float64)
    act = np.zeros((size, size), dtype=np.float64)
    cost_agt = np.zeros(d1.shape)
    cost_exp = np.zeros(d2.shape)
    ndim, nact = env.observation_space.shape[0], env.action_space.shape[0]

    cost = np.zeros(d1.shape)
    for i in range(d1.shape[0]):
        for j in range(d1.shape[1]):
            iobs = np.zeros(ndim)
            iobs[draw_dim[0]], iobs[draw_dim[1]] = d1[i][j], d2[i][j]
            ipacts, _ = algo.predict(np.array(iobs), deterministic=True)
            pact[i][j] = ipacts[draw_dim[2]]
            iacts, _ = exp.predict(iobs, deterministic=True)
            act[i][j] = iacts[draw_dim[2]]
            inp = torch.from_numpy(np.append(iobs, ipacts)).double()
            cost_agt[i][j] = costfn(inp).item()
            cost_exp[i][j] = iobs @ exp.Q @ iobs.T + iacts @ exp.R @ iacts.T * exp.gear**2

    cost_agt /= np.amax(cost_agt)
    cost_exp /= np.amax(cost_exp)
    title_list = ["norm_cost", "norm_cost", "abs_action", "abs_action"]
    yval_list = [cost_agt, cost_exp, np.abs(pact), np.abs(act)]
    xlabel, ylabel = dim_names[draw_dim[0]], dim_names[draw_dim[1]]
    max_list = [1.0, 1.0, 0.5, 0.5]
    min_list = [0.0, 0.0, 0.0, 0.0]
    fig = plt.figure()
    for i in range(4):
        ax = fig.add_subplot(2, 2, (i+1))
        surf = ax.pcolor(d1, d2, yval_list[i], cmap=cm.coolwarm,
                         shading='auto', vmax=max_list[i], vmin=min_list[i])
        clb = fig.colorbar(surf, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        clb.ax.set_title(title_list[i])
    plt.show()
