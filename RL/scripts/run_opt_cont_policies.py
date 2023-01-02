import pickle
import numpy as np
from scipy import io, signal
from imitation.data.rollout import make_sample_until, generate_trajectories
from imitation.data.types import Trajectory
from matplotlib import pyplot as plt

from RL.src import IDPLQRPolicy, IDPDiffLQRPolicy
from common.util import make_env


def draw_time_trajectories():
    # sample_until = make_sample_until(n_timesteps=None, n_episodes=5)
    # trajs = generate_trajectories(agent, agent.env, sample_until, deterministic_policy=True)
    trajs = []
    for traj in expt_trajs:
        ob = traj.obs[0]
        obs, acts, _ = agent.predict(ob, deterministic=True)
        trajs.append(Trajectory(obs=obs, acts=acts, infos=None))

    fig = plt.figure()
    for i in range(3):
        fig.add_subplot(3, 1, i+1)
    # for traj in expt_trajs:
    #     for i in range(2):
    #         fig.axes[i].plot(traj.obs[:, i], color='b')
    #     for j in range(1):
    #         fig.axes[j + 2].plot(traj.acts[:, j], color='b')
    for traj in trajs:
        for i in range(2):
            fig.axes[i].plot(traj.obs[:, i], color='k')
        for j in range(1):
            fig.axes[j + 2].plot(traj.acts[:, j], color='k')
    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    subj = "sub05"
    bsp = io.loadmat(f"../../IRL/demos/HPC/{subj}_full/{subj}i1.mat")['bsp']
    with open(f"../../IRL/demos/HPC/full/{subj}_1.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    init_states = []
    pltqs = []
    for traj in expt_trajs:
        init_states.append(traj.obs[0])
        pltqs.append(traj.pltq)
    env = make_env("HPC_custom-v0", bsp=bsp, init_states=init_states, pltqs=pltqs)
    agent = IDPDiffLQRPolicy(env, gamma=1.0, alpha=0.01)
    # print(agent.K)
    draw_time_trajectories()
    # clsys = signal.StateSpace(agent.A + agent.B @ agent.kks[0].numpy(), agent.B, np.eye(4), np.zeros([4, 2]), dt=agent.dt)
    # print(clsys.poles)
