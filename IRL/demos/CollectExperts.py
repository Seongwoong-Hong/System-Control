import os
import pickle
import torch as th
from scipy import io

from imitation.data import rollout, types

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.wrappers import *
from common.rollouts import generate_trajectories_without_shuffle


def main(actuation=1):
    # env_op = 0.1
    n_episodes = 30000
    env_type = "DiscretizedHuman"
    name = f"{env_type}"
    subj = "sub06"
    wrapper = ActionWrapper if "HPC" in env_type else None
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    subpath = os.path.join(proj_path, "IRL", "demos", "HPC", subj, subj)
    with open(f"{proj_path}/IRL/demos/{env_type}/19171717_done/{subj}_{actuation}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    init_states = []
    for traj in expt_trajs:
        init_states += [traj.obs[0]]
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    venv = make_env(f"{name}-v0", num_envs=1, wrapper=DiscretizeWrapper,
                    N=[19, 17, 17, 17], NT=[11, 11], bsp=bsp, init_states=init_states)
    # venv = make_env(env_name=f"{name}-v2", num_envs=1)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=len(init_states))
    with open(f"{proj_path}/RL/{env_type}/tmp/log/{name}_{subj}_19171717_done/softqiter/policies_1/agent.pkl",
              "rb") as f:
        ExpertPolicy = pickle.load(f)
    # with open(f"{proj_path}/IRL/tmp/log/{name}/MaxEntIRL/ext_viter_disc_linear_svm_reset/model/000/agent.pkl", "rb") as f:
    #     ExpertPolicy = pickle.load(f)
    # ExpertPolicy = PPO.load(f"{proj_path}/RL/{env_type}/tmp/log/{name}/ppo/policies_1/agent.pkl")
    # ExpertPolicy = PPO.load(f"{proj_path}/IRL/tmp/log/{name}/MaxEntIR L/ext_ppo_disc_samp_linear_ppoagent_svm_reset/model/000/agent")
    trajectories = generate_trajectories_without_shuffle(ExpertPolicy, venv, sample_until, deterministic_policy=False)
    save_name = f"{env_type}/19171717_disc_quadcost/{subj}_{actuation}.pkl"
    types.save(save_name, trajectories)
    print(f"Expert Trajectories are saved in the {save_name}")


if __name__ == "__main__":
    for actuation in range(1, 7):
        main(actuation)
