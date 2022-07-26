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

def test_expt_cost():
    env_type = "DiscretizedPendulum"
    env_id = f"{env_type}"
    with open("../../IRL/demos/DiscretizedPendulum/databased_lqr/obs_info_tree_300.pkl", "rb") as f:
        obs_info_tree = pickle.load(f)
    with open("../../IRL/demos/DiscretizedPendulum/databased_lqr/acts_info_tree_50.pkl", "rb") as f:
        acts_info_tree = pickle.load(f)
    obs_info = DataBasedDiscretizationInfo([0.05, 0.3], [-0.05, -0.08], obs_info_tree)
    acts_info = DataBasedDiscretizationInfo([30], [-40], acts_info_tree)
    venv = make_env(f"{env_id}-v2", num_envs=1, obs_info=obs_info, acts_info=acts_info)
    agent1 = DiscIPLQRPolicy(venv)
    sample_until = make_sample_until(n_timesteps=None, n_episodes=100)
    venv.reset()
    trajs1 = generate_trajectories(agent1, venv, sample_until=sample_until, deterministic_policy=True)
    init_states = []
    rew1 = 0
    for traj in trajs1:
        init_states.append(traj.obs[0])
        rew1 += traj.rews.sum()
    rew1 /= len(trajs1)
    agent2 = FiniteSoftQiter(venv, gamma=1, alpha=0.0001, device='cpu')
    agent2.learn(0)
    trajs2 = []
    rew2 = 0
    for i in range(len(init_states)):
        init_state = init_states[i]
        obs, acts, rews = agent2.predict(init_state, deterministic=True)
        data_dict = {'obs': obs, 'acts': acts, 'rews': rews.flatten(), 'infos': None}
        traj = types.TrajectoryWithRew(**data_dict)
        trajs2.append(traj)
        rew2 += rews.sum()
    rew2 /= len(init_states)
    print(rew1, rew2)
