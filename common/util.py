import os
import os.path as p
import gym
import gym_envs  # needs for custom environments

from scipy import io
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_env(env_name, use_vec_env=False, num_envs=10, use_norm=False, **kwargs):
    def is_vectorized(env):
        if use_norm:
            stats_path = kwargs.pop("stats_path", None)
            env = DummyVecEnv([lambda: env for _ in range(num_envs)])
            if not stats_path:
                env = VecNormalize(env, norm_obs=True, norm_reward=True)
            else:
                env = VecNormalize.load(stats_path, env)
        elif use_vec_env:
            env = DummyVecEnv([lambda: env for _ in range(num_envs)])
        return env

    if isinstance(env_name, gym.Env):
        env = env_name
        env_type = "Env"
    else:
        env_type = env_name[:env_name.find("_custom")]
        env = gym.make(env_name)
    if env_type == "HPC":
        subpath = kwargs.pop("subpath", None)
        pltqs = kwargs.get("pltqs")
        assert subpath is not None or pltqs is not None, "HPC environment needs pltqs!"
        if not pltqs:
            pltqs = []
            i = 0
            while True:
                file = subpath + f"i{i+1}.mat"
                if not p.isfile(file):
                    break
                pltqs += [io.loadmat(file)['pltq']]
                i += 1
            kwargs['pltqs'] = pltqs
    else:
        kwargs.pop('subpath', None)
        kwargs.pop('pltqs', None)
        kwargs.pop('bsp', None)
    return is_vectorized(env)


def write_analyzed_result(ana_fn,
                          ana_dir,
                          iter_name=None,
                          result_path: str = "/model/result.txt",
                          verbose=0):
    if iter_name is not None:
        filename = ana_dir + f"/{iter_name}" + result_path
        assert p.isdir(p.abspath(p.join(filename, p.pardir))), "Directory doesn't exist"
    else:
        filename = ana_dir + result_path
        assert p.isdir(p.abspath(p.join(filename, p.pardir))), "Directory doesn't exist"
    if not p.isfile(filename):
        ana_dict = ana_fn()
        f = open(filename + ".tmp", "w")
        for key, value in ana_dict.items():
            f.write(f"{key}: {value}\n")
            if verbose == 1:
                print(f"{key}: {value}")
        f.close()
        os.replace(filename + ".tmp", filename)
        if verbose == 1:
            print("The result file saved successfully")
    else:
        if verbose == 1:
            print("A result file already exists")


def remove_analyzed_result(ana_dir, entire=True, folder_num=None):
    if entire:
        folder_num = 0
        while True:
            file = ana_dir + f"/{folder_num}/model/result.txt"
            if not p.isdir(p.dirname(file)):
                print(f"Break at the folder #{folder_num}")
                break
            if not p.isfile(file):
                print(f"Pass the folder #{folder_num}")
                folder_num += 1
                continue
            os.remove(file)
            folder_num += 1
    else:
        assert folder_num is not None, "The folder number was not specified"
        file = ana_dir + f"/{folder_num}/model/result.txt"
        if p.isfile(file):
            os.remove(file)
            print(f"The result file in the folder #{folder_num} is removed successfully")
