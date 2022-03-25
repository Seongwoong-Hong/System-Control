import os
import pickle
import torch as th
from scipy import io

from imitation.data import rollout, types

from algos.tabular.viter import FiniteSoftQiter
from algos.torch.sac import SAC
from common.util import make_env
from common.wrappers import *
from common.rollouts import generate_trajectories_without_shuffle


def main(actuation=1):
    # env_op = 0.1
    n_episodes = 16
    env_type = "2DTarget"
    name = f"{env_type}"
    subj = "sub06"
    wrapper = ActionWrapper if "HPC" in env_type else None
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    subpath = os.path.join(proj_path, "IRL", "demos", "HPC", subj, subj)
    with open(f"{proj_path}/IRL/demos/DiscretizedHuman/17171719/{subj}_{actuation}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    init_states = []
    for traj in expt_trajs:
        init_states += [traj.obs[0]]
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    # venv = make_env(f"{name}-v2", num_envs=1, bsp=bsp)#, init_states=init_states)
    venv = make_env(f"{name}-v2", num_envs=1)

    # ExpertPolicy = FiniteSoftQiter(venv, gamma=1, alpha=0.001, device='cpu')
    # ExpertPolicy.learn(0)
    # trajectories = []
    # for _ in range(17*17*17*19):
    #     init_state = venv.reset()[0]
    #     obs, acts, rews = ExpertPolicy.predict(init_state, deterministic=True)
    #     data_dict = {'obs': obs, 'acts': acts, 'rews': rews.flatten(), 'infos': None}
    #     traj = types.TrajectoryWithRew(**data_dict)
    #     trajectories.append(traj)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    ExpertPolicy = SAC.load(f"{proj_path}/RL/{env_type}/tmp/log/{name}/sac/policies_1/agent.zip")
    # with open(f"{proj_path}/RL/{env_type}/tmp/log/{name}_{subj}_customshape/softqiter/policies_2/agent.pkl", "rb") as f:
    #     ExpertPolicy = pickle.load(f)
    trajectories = generate_trajectories_without_shuffle(ExpertPolicy, venv, sample_until, deterministic_policy=False)
    save_name = f"{env_type}/sac_1.pkl"
    types.save(save_name, trajectories)
    print(f"Expert Trajectories are saved in the {save_name}")


if __name__ == "__main__":
    for actuation in range(1, 2):
        main(actuation)
