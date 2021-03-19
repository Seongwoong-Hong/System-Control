import gym_envs
from imitation.data import rollout, types
from stable_baselines3.common.vec_env import DummyVecEnv

from IRL.project_policies import def_policy

if __name__ == "__main__":
    n_steps, n_episodes = 900, 20
    env_type = "HPC"
    env_name = "{}_custom-v0".format(env_type)
    env = gym_envs.make(env_name, n_steps=n_steps)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    ExpertPolicy = def_policy(env_type, env)
    trajectories = rollout.generate_trajectories(
        ExpertPolicy, DummyVecEnv([lambda: env]), sample_until, deterministic_policy=False)
    save_name = "demos/{}/expert.pkl".format(env_type)
    types.save(save_name, trajectories)
    print("Expert Trajectories are saved")
