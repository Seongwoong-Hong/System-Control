import os

from imitation.data import rollout, types
from stable_baselines3.common.vec_env import DummyVecEnv

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.wrappers import ActionWrapper
from common.rollouts import generate_trajectories_without_shuffle
from IRL.scripts.project_policies import def_policy

if __name__ == "__main__":
    n_episodes = 110
    env_type = "2DWorld"
    name = f"{env_type}"
    subpath = "HPC/sub01/sub01"
    env = make_env(env_name=f"{name}-v0", use_vec_env=False, subpath=subpath, wrapper=ActionWrapper)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # ExpertPolicy = def_policy(env_type, env, noise_lv=0.25)
    ExpertPolicy = SAC.load(f"{proj_path}/../RL/{env_type}/tmp/log/{name}/sac/policies_1/agent.zip")
    # ExpertPolicy = PPO.load(f"{proj_path}/tmp/log/IDP/ppo/lqrppo/000000500000/model.pkl")
    trajectories = generate_trajectories_without_shuffle(
        ExpertPolicy, DummyVecEnv([lambda: env]), sample_until, deterministic_policy=False)
    save_name = f"{env_type}/sac.pkl"
    types.save(save_name, trajectories)
    print(f"Expert Trajectories are saved in the {save_name}")
