import gym_envs
import os
from imitation.data import rollout, types
from stable_baselines3.common.vec_env import DummyVecEnv

from algo.torch.ppo import PPO
from common.util import make_env
from IRL.project_policies import def_policy

if __name__ == "__main__":
    n_steps, n_episodes = 600, 35
    env_type = "HPC"
    sub = "sub01"
    env = make_env(env_name=f"{env_type}_custom-v0", use_vec_env=False, sub=sub, n_steps=n_steps)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    ExpertPolicy = def_policy(env_type, env)
    # ExpertPolicy = PPO.load("tmp/log/HPC/ppo/2021-3-22-13-48-51/model/extra_ppo1.zip")
    trajectories = rollout.generate_trajectories(
        ExpertPolicy, DummyVecEnv([lambda: env]), sample_until, deterministic_policy=False)
    save_name = f"demos/{env_type}/lqrTest.pkl"
    types.save(save_name, trajectories)
    print(f"Expert Trajectories are saved in the {save_name}")
