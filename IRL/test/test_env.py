import gym
import gym_envs
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from common.modules import NNCost
from common.wrappers import CostWrapper

if __name__ == "__main__":
    n_steps = 100
    device = 'cpu'
    env_id = "IP_custom-v1"
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    inp = num_obs + num_act
    costfn = NNCost(arch=[inp], device=device, num_samp=5).double().to(device)
    env = DummyVecEnv([lambda: CostWrapper(env, costfn)])
    env.step(np.array([0.1]))
    print("test")
