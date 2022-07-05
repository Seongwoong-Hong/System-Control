import os
import pickle
import json
import torch as th
from scipy import io

from imitation.data import rollout, types

from algos.tabular.viter import FiniteSoftQiter
from algos.torch.sac import SAC
from algos.torch.ppo import PPO
from algos.torch.OptCont import LQRPolicy
from common.util import make_env
from common.wrappers import *
from common.rollouts import generate_trajectories_without_shuffle


class IDPLQRPolicy(LQRPolicy):
    def _build_env(self) -> np.array:
        I1, I2 = 0.878121, 1.047289
        l1 = 0.7970
        lc1, lc2 = 0.5084, 0.2814
        m1 ,m2 = 17.2955, 34.5085
        g = 9.81
        M = np.array([[I1 + m1*lc1**2 + I2 + m2*l1**2 + 2*m2*l1*lc2 + m2*lc2**2, I2 + m2*l1*lc2 + m2*lc2**2],
                      [I2 + m2*l1*lc2 + m2*lc2**2, I2 + m2*lc2**2]])
        C = np.array([[m1*lc1*g + m2*l1*g + m2*g*lc2, m2*g*lc2],
                      [m2*g*lc2, m2*g*lc2]])
        self.A, self.B = np.zeros([4, 4]), np.zeros([4, 2])
        self.A[:2, 2:] = np.eye(2, 2)
        self.A[2:, :2] = np.linalg.inv(M) @ C
        self.B[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
        self.Q = np.diag([3.5139, 0.2872182, 0.24639979, 0.01540204])
        self.R = np.diag([0.02537065/1600, 0.01358577/900])
        self.gear = 100

def main():
    # env_op = 0.1
    env_type = "IDP"
    name = f"{env_type}_custom"
    wrapper = ActionWrapper if "HPC" in env_type else None
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
    # env = make_env(f"{name}-v0", wrapper=DiscretizeWrapper)

    # ExpertPolicy = FiniteSoftQiter(env, gamma=1, alpha=0.001, device='cpu')
    # ExpertPolicy.learn(0)
    # trajectories = []
    # for _ in range(300):
    # for _ in range(len(init_states)):
        # init_state = env.reset()
        # obs, acts, rews = ExpertPolicy.predict(init_state, deterministic=False)
        # data_dict = {'obs': obs, 'acts': acts, 'rews': rews.flatten(), 'infos': None}
        # traj = types.TrajectoryWithRew(**data_dict)
        # trajectories.append(traj)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=len(init_states))
    # ExpertPolicy = PPO.load(f"{proj_path}/RL/{env_type}/tmp/log/{name}/ppo/policies_1/agent_10.zip")
    ExpertPolicy = IDPLQRPolicy(env=make_env(f"{name}-v2"))
    # with open(f"{proj_path}/RL/{env_type}/tmp/log/{name}_{subj}_customshape/softqiter/policies_2/agent.pkl", "rb") as f:
    #     ExpertPolicy = pickle.load(f)
    # venv = make_env(f"{name}-v0", num_envs=1, bsp=bsp, init_states=init_states)
    venv = make_env(f"{name}-v0", num_envs=1, init_states=init_states)
    trajectories = generate_trajectories_without_shuffle(ExpertPolicy, venv, sample_until, deterministic_policy=True)
    # save_name = f"{env_type}/cont_quadcost/{subj}_{actuation}.pkl"
    save_name = f"{env_type}/lqr_sub05.pkl"
    types.save(save_name, trajectories)

    print(f"Expert Trajectories are saved in the {save_name}")


if __name__ == "__main__":
    for subj in [f"sub{i:02d}" for i in [5]]:
        for actuation in range(4, 5):
            main()
