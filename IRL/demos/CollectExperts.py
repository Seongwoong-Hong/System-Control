import os
import pickle
import json
import torch as th
from scipy import io

from imitation.data import rollout, types

from algos.tabular.viter import FiniteSoftQiter
from algos.torch.sac import SAC
from common.util import make_env
from common.wrappers import *
from common.rollouts import generate_trajectories_without_shuffle


def main():
    # env_op = 0.1
    n_episodes = 1
    env_type = "2DWorld"
    name = f"{env_type}_disc"
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
    # env = make_env(f"{name}-v0", bsp=bsp, N=[19, 19, 19, 19], NT=[11, 11], init_states=init_states, wrapper=DiscretizeWrapper)
    # perturbation = actuation - 1
    # max_states = bound_info[subj][perturbation]["max_states"]
    # min_states = bound_info[subj][perturbation]["min_states"]
    # max_torques = bound_info[subj][perturbation]["max_torques"]
    # min_torques = bound_info[subj][perturbation]["min_torques"]
    # env.set_bounds(max_states, min_states, max_torques, min_torques)
    env = make_env(f"{name}-v0", wrapper=DiscretizeWrapper)

    ExpertPolicy = FiniteSoftQiter(env, gamma=1, alpha=0.001, device='cpu')
    ExpertPolicy.learn(0)
    trajectories = []
    # for _ in range(len(expt_trajs)*20):
    from matplotlib import pyplot as plt
    for _ in range(440):
        init_state = env.reset()
        obs, acts, rews = ExpertPolicy.predict(init_state, deterministic=False)
        data_dict = {'obs': obs, 'acts': acts, 'rews': rews.flatten(), 'infos': None}
        traj = types.TrajectoryWithRew(**data_dict)
        trajectories.append(traj)
    # sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=len(init_states))
    # ExpertPolicy = SAC.load(f"{proj_path}/RL/{env_type}/tmp/log/{name}/sac/policies_1/agent.zip")
    # with open(f"{proj_path}/RL/{env_type}/tmp/log/{name}_{subj}_customshape/softqiter/policies_2/agent.pkl", "rb") as f:
    #     ExpertPolicy = pickle.load(f)
    # trajectories = generate_trajectories_without_shuffle(ExpertPolicy, venv, sample_until, deterministic_policy=False)
    save_name = f"{env_type}/dot2_1dot2_8_largestate/001alpha_nobias.pkl"
    types.save(save_name, trajectories)
    print(f"Expert Trajectories are saved in the {save_name}")


if __name__ == "__main__":
    for subj in [f"sub{i:02d}" for i in [5]]:
        for actuation in range(1, 2):
            main()
