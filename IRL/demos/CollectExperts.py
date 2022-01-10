import os
import pickle
import torch as th
from scipy import io

from imitation.data import rollout, types
from stable_baselines3.common.vec_env import DummyVecEnv

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.wrappers import ActionWrapper
from common.rollouts import generate_trajectories_without_shuffle
from IRL.scripts.project_policies import def_policy

if __name__ == "__main__":
    # env_op = 0.1
    n_episodes = 100
    env_type = "2DTarget_disc"
    name = f"{env_type}"
    subj = "sub07"
    wrapper = ActionWrapper if "HPC" in env_type else None
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    subpath = os.path.join(proj_path, "HPC", f"{subj}_cropped", subj)
    init_states = []
    for i in range(1):
        for j in range(6):
            bsp = io.loadmat(subpath + f"i{i + 1}_{j}.mat")['bsp']
            init_states += [io.loadmat(subpath + f"i{i + 1}_{j}.mat")['state'][0, :4]]
    # venv = make_env(env_name=f"{name}-v0", num_envs=1, subpath=subpath, wrapper=wrapper, N=[11, 17, 17, 17])
    venv = make_env(env_name=f"{name}-v2", num_envs=1)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(f"{proj_path}/../RL/{env_type}/tmp/log/{name}_more_random/softqiter/policies_1/agent.pkl", "rb") as f:
        ExpertPolicy = pickle.load(f)
    # with open(f"{proj_path}/tmp/log/{name}/MaxEntIRL/ext_viter_disc_linear_svm_reset/model/000/agent.pkl", "rb") as f:
    #     ExpertPolicy = pickle.load(f)
    # ExpertPolicy = PPO.load(f"{proj_path}/../RL/{env_type}/tmp/log/{name}/ppo/policies_1/agent.pkl")
    # ExpertPolicy = PPO.load(f"{proj_path}/tmp/log/{name}/MaxEntIR L/ext_ppo_disc_samp_linear_ppoagent_svm_reset/model/000/agent")
    trajectories = generate_trajectories_without_shuffle(ExpertPolicy, venv, sample_until, deterministic_policy=False)
    save_name = f"{env_type}/50/softqiter.pkl"
    types.save(save_name, trajectories)
    print(f"Expert Trajectories are saved in the {save_name}")
