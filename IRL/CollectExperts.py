import gym_envs
import os
from imitation.data import rollout, types
from stable_baselines3.common.vec_env import DummyVecEnv

from algo.torch.ppo import PPO
from IRL.project_policies import def_policy
from scipy import io

if __name__ == "__main__":
    n_steps, n_episodes = 600, 35
    env_type = "HPC"
    sub = "sub01"
    pltqs = []
    if env_type == "HPC":
        for i in range(35):
            file = os.path.join("demos", env_type, sub, sub + "i%d.mat" % (i + 1))
            pltqs += [io.loadmat(file)['pltq']]
        env = gym_envs.make("{}_custom-v0".format(env_type), n_steps=n_steps, pltqs=pltqs)
    else:
        env = gym_envs.make("{}_custom-v0".format(env_type), n_steps=n_steps)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    ExpertPolicy = def_policy(env_type, env)
    # ExpertPolicy = PPO.load("tmp/log/HPC/ppo/2021-3-22-13-48-51/model/extra_ppo1.zip")
    trajectories = rollout.generate_trajectories(
        ExpertPolicy, DummyVecEnv([lambda: env]), sample_until, deterministic_policy=False)
    save_name = "demos/{}/lqrTest.pkl".format(env_type)
    types.save(save_name, trajectories)
    print("Expert Trajectories are saved")
