import os
import pickle
import json
import torch as th
from scipy import io

from imitation.data import rollout, types

from algos.tabular.viter import FiniteSoftQiter, SoftQiter
from algos.torch.sac import SAC
from algos.torch.ppo import PPO
from algos.torch.OptCont import LQRPolicy
from common.util import make_env
from common.wrappers import *
from common.rollouts import generate_trajectories_without_shuffle


class IPLQRPolicy(LQRPolicy):
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


def main():
    # env_op = 0.1
    env_type = "DiscretizedPendulum"
    name = f"{env_type}_DataBased"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    subpath = os.path.join(proj_path, "IRL", "demos", "HPC", subj, subj)
    with open(f"{proj_path}/IRL/demos/DiscretizedHuman/19191919/{subj}_{actuation}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    with open(f"{proj_path}/IRL/demos/bound_info.json", "r") as f:
        bound_info = json.load(f)
    init_states = []
    for traj in expt_trajs:
        init_states += [traj.obs[0]]
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    # env = make_env(f"{name}-v0", bsp=bsp, N=[19, 19, 19, 19], NT=[11, 11], wrapper=DiscretizeWrapper, init_states=init_states)
    # perturbation = actuation - 1
    # max_states = bound_info[subj][perturbation]["max_states"]
    # min_states = bound_info[subj][perturbation]["min_states"]
    # max_torques = bound_info[subj][perturbation]["max_torques"]
    # min_torques = bound_info[subj][perturbation]["min_torques"]
    # env.set_bounds(max_states, min_states, max_torques, min_torques)
    with open("../../tests/envs/obs_test.pkl", "rb") as f:
        obs_info_tree = pickle.load(f)
    with open("../../tests/envs/acts_test.pkl", "rb") as f:
        acts_info_tree = pickle.load(f)
    venv = make_env(f"{name}-v2", num_envs=1, N=[29, 29], NT=[21], wrapper=DiscretizeWrapper)
    venv.envs[0].obs_info.info_tree = obs_info_tree
    # venv.envs[0].acts_info.info_tree = acts_info_tree

    # ExpertPolicy = FiniteSoftQiter(venv, gamma=1, alpha=0.001, device='cpu')
    # ExpertPolicy = SoftQiter(venv, gamma=0.999, alpha=0.001, device='cpu')
    # ExpertPolicy.learn(0)
    # trajectories = []
    # for _ in range(5000):
    # for _ in range(len(init_states)):
    #     init_state = venv.reset()
    #     obs, acts, rews = ExpertPolicy.predict(init_state, deterministic=True)
    #     data_dict = {'obs': obs, 'acts': acts, 'rews': rews.flatten(), 'infos': None}
    #     traj = types.TrajectoryWithRew(**data_dict)
    #     trajectories.append(traj)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=300)
    # ExpertPolicy = PPO.load(f"{proj_path}/RL/{env_type}/tmp/log/{name}/ppo/policies_1/agent_10.zip")
    ExpertPolicy = IPLQRPolicy(env=make_env(f"{name}-v2", N=[29, 29], NT=[21]))
    # venv = make_env(f"{name}-v2", num_envs=1, N=[19, 19], NT=[11])
    trajectories = generate_trajectories_without_shuffle(ExpertPolicy, venv, sample_until, deterministic_policy=True)
    # save_name = f"{env_type}/cont_quadcost/{subj}_{actuation}.pkl"
    save_name = f"{env_type}/databased_21_lqr/quadcost_lqr.pkl"
    types.save(save_name, trajectories)

    print(f"Expert Trajectories are saved in the {save_name}")


if __name__ == "__main__":
    for subj in [f"sub{i:02d}" for i in [5]]:
        for actuation in range(4, 5):
            main()
