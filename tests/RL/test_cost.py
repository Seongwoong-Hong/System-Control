import os
import pickle
import numpy as np
import torch as th
from matplotlib import pyplot as plt

from imitation.data.rollout import make_sample_until, generate_trajectories, flatten_trajectories, types

from gym_envs.envs import DataBasedDiscretizationInfo
from algos.torch.OptCont import LQRPolicy
from algos.tabular.viter import FiniteSoftQiter
from common.util import make_env

class DiscIPLQRPolicy(LQRPolicy):
    def _build_env(self):
        g = 9.81
        m = 17.2955
        l = 0.7970
        lc = 0.5084
        I = 0.878121 + m * lc**2
        self.A, self.B = np.zeros([2, 2]), np.zeros([2, 1])
        self.A[0, 1] = 1
        self.A[1, 0] = m * g * lc / I
        self.B[1, 0] = 1 / I
        self.Q = np.diag([2.8139, 1.04872182])
        self.R = np.diag([1.617065e-4])
        self.gear = 1

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        action = super(DiscIPLQRPolicy, self)._predict(observation, deterministic)
        high = th.from_numpy(self.env.action_space.high).float()
        low = th.from_numpy(self.env.action_space.low).float()
        action = th.clip(action, min=low, max=high)
        d_act = self.env.env_method("get_acts_from_idx", self.env.env_method("get_idx_from_acts", action.numpy())[0])[0]
        return th.from_numpy(d_act).float()


def draw_trajs(traj1, traj2):
    fig = plt.figure(figsize=[6.4, 9.6])
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.scatter(range(50), traj1.obs[:-1, 0])
    ax1.scatter(range(50), traj2.obs[:-1, 0])
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.scatter(range(50), traj1.obs[:-1, 1])
    ax2.scatter(range(50), traj2.obs[:-1, 1])
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.scatter(range(50), traj1.acts)
    ax3.scatter(range(50), traj2.acts)
    plt.show()


def test_expt_cost(hpc_env, idpdiffpolicy):
    proj_path = os.path.abspath("../..")
    with open(f"{proj_path}/IRL/demos/HPC/full/sub01_1.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    agent = idpdiffpolicy(hpc_env, alpha=0.001, gamma=1)
    Q, R1, R2 = agent.Q, agent.R1, agent.R2
    cost1, cost2 = 0, 0
    for traj in expt_trajs:
        init_state = traj.obs[0]
        agent_obs, agent_acts, _ = agent.predict(init_state, deterministic=True)
        agent_prev_acts = np.append(np.zeros([1, 2]), agent_acts, axis=0)
        agent_diff_acts = agent_acts - agent_prev_acts[:-1]
        expt_diff_acts = traj.acts - np.append(np.zeros([1, 2]), traj.acts, axis=0)[:-1]
        cost1 += np.sum(
            np.sum((traj.obs[:-1] @ Q) * traj.obs[:-1], axis=1) +
            np.sum((traj.acts @ R1) * traj.acts, axis=1) +
            np.sum((expt_diff_acts @ R2) * expt_diff_acts, axis=1)
        )
        cost2 += np.sum(
            np.sum((agent_obs[:-1] @ Q) * agent_obs[:-1], axis=1) +
            np.sum((agent_acts @ R1) * agent_acts, axis=1) +
            np.sum((agent_diff_acts @ R2) * agent_diff_acts, axis=1)
        )
    cost1 /= len(expt_trajs)
    cost2 /= len(expt_trajs)
    print(cost1, cost2)
