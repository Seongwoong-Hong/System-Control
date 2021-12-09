import os
import warnings
import os.path as p
import gym
import gym_envs  # needs for custom environments
from copy import deepcopy

from scipy import io
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from common.wrappers import ActionWrapper


def make_env(env_name, num_envs=None, use_norm=False, wrapper=None, **kwargs):
    wrapper_kwargs = kwargs.pop('wrapper_kwrags', {})
    if isinstance(env_name, gym.Env):
        env = env_name
    else:
        if "HPC" in env_name:
            subpath = kwargs.pop("subpath", None)
            pltqs = kwargs.get("pltqs")
            if pltqs is None and subpath is not None:
                pltqs, init_states = [], []
                i = 0
                while True:
                    file = subpath + f"i{i + 1}.mat"
                    if not p.isfile(file):
                        break
                    pltqs += [io.loadmat(file)['pltq']]
                    init_states += [io.loadmat(file)['state'][0, :4]]
                    kwargs['bsp'] = io.loadmat(file)['bsp']
                    i += 1
                kwargs['pltqs'] = pltqs
                kwargs['init_states'] = init_states
        elif "Human" not in env_name:
            kwargs.pop('bsp', None)
        else:
            kwargs.pop('subpath', None)
            kwargs.pop('pltqs', None)
        env = gym.make(env_name, **kwargs)

    if wrapper is not None:
        if wrapper == "ActionWrapper":
            env = ActionWrapper(env)
        elif isinstance(wrapper, str):
            warnings.warn("Not specified wrapper name so it's ignored")
        else:
            env = wrapper(env, **wrapper_kwargs)

    if use_norm:
        env = DummyVecEnv([lambda: Monitor(deepcopy(env)) for _ in range(num_envs)])
        if isinstance(use_norm, str):
            env = VecNormalize.load(use_norm, env)
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
    elif num_envs:
        env = DummyVecEnv([lambda: deepcopy(env) for _ in range(num_envs)])

    return env


def write_analyzed_result(
        ana_fn,
        ana_dir,
        iter_name=None,
        result_path: str = "/model/result.txt",
        verbose=0
):
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
