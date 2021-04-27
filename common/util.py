import os
import gym_envs

from stable_baselines3.common.vec_env import DummyVecEnv
from scipy import io


def make_env(env_name, use_vec_env=True, num_envs=4, n_steps=None, sub=None):
    env_type = env_name[:3]
    if env_type == "HPC":
        pltqs = []
        i = 0
        while True:
            file = os.path.join("demos", env_type, sub, sub + "i%d.mat" % (i + 1))
            if not os.path.isfile(file):
                break
            pltqs += [io.loadmat(file)['pltq']]
            i += 1
        if use_vec_env:
            env = DummyVecEnv([lambda: gym_envs.make(env_name, n_steps=n_steps, pltqs=pltqs) for _ in range(num_envs)])
        else:
            env = gym_envs.make(env_name, n_steps=n_steps, pltqs=pltqs)
    else:
        if use_vec_env:
            env = DummyVecEnv([lambda: gym_envs.make(env_name, n_steps=n_steps) for _ in range(num_envs)])
        else:
            env = gym_envs.make(env_name, n_steps=n_steps)
    return env
