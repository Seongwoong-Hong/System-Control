import pickle
import numpy as np
import torch as th
from scipy import io
from matplotlib import pyplot as plt
from common.util import make_env
from RL.src import IDPLQRPolicy, IDPiterLQRPolicy


with open("../../IRL/demos/HPC/full/sub05_4.pkl", "rb") as f:
    expt_trajs = [pickle.load(f)[0]]
tg_obs = np.append(expt_trajs[0].obs, np.zeros([360, 4]), axis=0)
tg_acts = np.append(expt_trajs[0].acts, np.zeros([360, 2]), axis=0) * 300
perturb = np.append(expt_trajs[0].pltq, np.zeros([360, 2]), axis=0)
# perturb = np.zeros([720, 2])


class IDPLQRPolicyTime(IDPLQRPolicy):
    def _build_env(self):
        super(IDPLQRPolicyTime, self)._build_env()
        # self.q = np.zeros([720, 4])
        # self.r = np.zeros([720, 2])
        self.q = -tg_obs @ self.Q
        self.r = -tg_acts @ self.R


if __name__ == "__main__":
    bsp = io.loadmat("../../IRL/demos/HPC/sub05/sub05i1.mat")['bsp']
    init_states = []
    pltqs = []
    for traj in expt_trajs:
        init_states.append(traj.obs[0])
        pltqs.append(traj.pltq)

    env = make_env("HPC_custom-v2", num_envs=1, bsp=bsp, init_states=init_states, pltqs=pltqs)
    agent = IDPLQRPolicyTime(env, gamma=1, alpha=0.001)
    # agent.learn()
    fig = plt.figure()
    [fig.add_subplot(3, 2, i+1) for i in range(6)]
    for traj in expt_trajs:
        for i in range(4):
            fig.axes[i].plot(traj.obs[:, i], 'b')
        for j in range(2):
            fig.axes[j+4].plot(traj.acts[:, j], 'b')

    for _ in range(15):
        obs = env.reset()
        obs_list, acts_list, _ = agent.predict(obs, deterministic=True)
        # done = False
        # obs_list, acts_list = [], []
        # obs_list.append(obs.flatten())
        # while not done:
        #     act, _ = agent.predict(obs)
        #     obs, _, done, _ = env.step(act)
        #     obs_list.append(obs.flatten())
        #     acts_list.append(act.flatten())
        for i in range(4):
            fig.axes[i].plot(np.array(obs_list)[:, i], 'k')
        for j in range(2):
            fig.axes[j + 4].plot(np.array(acts_list)[:, j], 'k')
    fig.tight_layout()
    fig.show()
