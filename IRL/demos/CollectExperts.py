import os

from imitation.data import rollout, types
from stable_baselines3.common.vec_env import DummyVecEnv

from algos.torch.ppo import PPO
from common.util import make_env
from IRL.scripts.project_policies import def_policy

if __name__ == "__main__":
    n_steps, n_episodes = 600, 10
    env_type = "IDP"
    name = "IDP_custom"
    subpath = "HPC/sub01/sub01"
    env = make_env(env_name=f"{name}-v2", use_vec_env=False, subpath=subpath, n_steps=n_steps)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # ExpertPolicy = def_policy(env_type, env, noise_lv=0.5)
    ExpertPolicy = PPO.load(f"{proj_path}/../RL/{env_type}/tmp/log/{name}_abs_ppo/ppo/policies_2/ppo0.zip")
    # ExpertPolicy = PPO.load(f"{proj_path}/tmp/log/IDP/ppo/lqrppo/000000500000/model.pkl")
    trajectories = rollout.generate_trajectories(
        ExpertPolicy, DummyVecEnv([lambda: env]), sample_until, deterministic_policy=False)
    save_name = f"{env_type}/abs_ppo.pkl"
    types.save(save_name, trajectories)
    print(f"Expert Trajectories are saved in the {save_name}")
