import os
import pickle
import torch as th

from imitation.data import rollout, types
from stable_baselines3.common.vec_env import DummyVecEnv

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.wrappers import ActionWrapper
from common.rollouts import generate_trajectories_without_shuffle
from IRL.scripts.project_policies import def_policy

if __name__ == "__main__":
    def feature_fn(x):
        return th.cat([x, x**2], dim=1)
    map_size = 20
    n_episodes = 100
    env_type = "1DTarget"
    name = f"{env_type}_disc"
    subpath = "HPC/sub01/sub01"
    wrapper = ActionWrapper if "HPC" in env_type else None
    env = make_env(env_name=f"{name}-v0", use_vec_env=False, subpath=subpath, wrapper=wrapper, map_size=map_size)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(f"{proj_path}/../RL/{env_type}/tmp/log/{name}_{map_size}/softqlearning/policies_1/agent.pkl", "rb") as f:
        ExpertPolicy = pickle.load(f)
    # with open(f"{proj_path}/tmp/log/{name}/MaxEntIRL/ext_viter_disc_linear_svm_reset/model/000/agent.pkl", "rb") as f:
    #     ExpertPolicy = pickle.load(f)
    # ExpertPolicy = PPO.load(f"{proj_path}/../RL/{env_type}/tmp/log/{name}/ppo/policies_1/agent.pkl")
    # ExpertPolicy = PPO.load(f"{proj_path}/tmp/log/{name}/MaxEntIRL/ext_ppo_disc_samp_linear_ppoagent_svm_reset/model/000/agent")
    trajectories = generate_trajectories_without_shuffle(
        ExpertPolicy, DummyVecEnv([lambda: env]), sample_until, deterministic_policy=False)
    save_name = f"{env_type}/softqlearning_disc_{map_size}.pkl"
    types.save(save_name, trajectories)
    print(f"Expert Trajectories are saved in the {save_name}")
