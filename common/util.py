import os
import gym_envs

from stable_baselines3.common.vec_env import DummyVecEnv
from scipy import io


def make_env(env_name, use_vec_env=True, num_envs=10, sub=None, **kwargs):
    env_type = env_name[:3]
    if env_type == "HPC":
        if sub is not None:
            pltqs = []
            i = 0
            while True:
                file = os.path.join("demos", env_type, sub, sub + "i%d.mat" % (i + 1))
                if not os.path.isfile(file):
                    break
                pltqs += [io.loadmat(file)['pltq']]
                i += 1
        if use_vec_env:
            env = DummyVecEnv([lambda: gym_envs.make(env_name, **kwargs) for _ in range(num_envs)])
        else:
            env = gym_envs.make(env_name, **kwargs)
    else:
        kwargs.pop('pltqs', None)
        if use_vec_env:
            env = DummyVecEnv([lambda: gym_envs.make(env_name, **kwargs) for _ in range(num_envs)])
        else:
            env = gym_envs.make(env_name, **kwargs)
    return env


def remove_analyzed_result(ana_dir, entire=True, folder_num=None):
    if entire:
        folder_num = 0
        while True:
            file = ana_dir + f"/{folder_num}/model/result.txt"
            if not os.path.isdir(os.path.dirname(file)):
                print(f"Break at the folder #{folder_num}")
                break
            if not os.path.isfile(file):
                print(f"Pass the folder #{folder_num}")
                folder_num += 1
                continue
            os.remove(file)
            folder_num += 1
    else:
        assert folder_num is not None, "The folder number was not specified"
        file = ana_dir + f"/{folder_num}/model/result.txt"
        if os.path.isfile(file):
            os.remove(file)
            print(f"The result file in the folder #{folder_num} is removed successfully")
