import copy
import inspect
import os
import torch
import pickle
import warnings
import os.path as p
import gym
import gym_envs  # needs for custom environments
from copy import deepcopy
from io import BytesIO

from scipy import io
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from common.wrappers import ActionWrapper


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def make_env(env_name, num_envs=None, use_norm=False, wrapper=None, **kwargs):
    wrapper_kwargs = kwargs.pop('wrapper_kwargs', {})

    def define_env():
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
            # elif "Human" not in env_name:
            #     kwargs.pop('bsp', None)
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
        return env

    if use_norm:
        if num_envs is None:
            num_envs = 1
        env = DummyVecEnv([lambda: Monitor(define_env()) for _ in range(num_envs)])
        if isinstance(use_norm, str):
            env = VecNormalize.load(use_norm, env)
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
    elif num_envs:
        env = DummyVecEnv([lambda: Monitor(define_env()) for _ in range(num_envs)])
    else:
        env = define_env()

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

# code from Hyunho Jeong

import collections
import random
import re
from functools import reduce

import numpy as np
import torch
from colorama import Fore, Style


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def extract_number(s):
    match = re.search(r'(\d+)', s)
    if match:
        return int(match.group(1))
    return None


def batch_by_window(t: torch.Tensor, w_size: int) -> torch.Tensor:
    """ make batch tensor by shifting every window of size 'w_size' """

    time_length = t.size(0)
    assert t.ndim == 2, f"Tensor of (T, D) is expected, but got {t.shape}"
    assert time_length >= w_size, 'too short time length'

    net_in = torch.stack([t[i:i+w_size, :] for i in range(time_length - w_size + 1)], dim=0)

    return net_in


def yes_or_no(msg) -> bool:
    """ query input and return true if yes else false """

    while True:
        cont = input(Fore.GREEN + f"{msg}, yes/no > ")

        # loop for different answer
        while cont.lower() not in ("yes", "no"):
            cont = input(Fore.GREEN + f"{msg}, yes/no > ")

        print(Style.RESET_ALL, end='')

        if cont == "yes":
            return True
        else:
            return False


def sample_config(config: dict) -> dict:
    sampled = {}

    for k, v in config.items():
        if not hasattr(v, '__iter__') or type(v) is str:
            sampled[k] = v
        elif type(v) is list:
            sampled[k] = random.choice(v)
        elif type(v) is dict:
            if k in sampled.values():
                sub_sampled = sample_config(v)
                sampled[k] = sub_sampled
        else:
            raise ValueError(f"unexpected type of configuration domain: ({k}:{v})")

    return sampled


def dot_map_dict_to_nested_dict(dot_map_dict):
    """Convert something like
    ```
    {
        'one.two.three.four': 4,
        'one.six.seven.eight': None,
        'five.nine.ten': 10,
        'five.zero': 'foo',
    }
    ```
    into its corresponding nested dict.
    http://stackoverflow.com/questions/16547643/convert-a-list-of-delimited-strings-to-a-tree-nested-dict-using-python
    """
    tree = {}

    for key, item in dot_map_dict.items():
        split_keys = key.split('.')
        if len(split_keys) == 1:
            if key in tree:
                raise ValueError("Duplicate key: {}".format(key))
            tree[key] = item
        else:
            t = tree
            for sub_key in split_keys[:-1]:
                t = t.setdefault(sub_key, {})
            last_key = split_keys[-1]
            if not isinstance(t, dict):
                raise TypeError(
                    "Key inside dot map must point to dictionary: {}".format(
                        key
                    )
                )
            if last_key in t:
                raise ValueError("Duplicate key: {}".format(last_key))
            t[last_key] = item

    return tree


def nested_dict_to_dot_map_dict(d, parent_key=''):
    """
    Convert a recursive dictionary into a flat, dot-map dictionary.

    :param d: e.g. {'a': {'b': 2, 'c': 3}}
    :param parent_key: Used for recursion
    :return: e.g. {'a.b': 2, 'a.c': 3}
    """
    items = []

    for k, v in d.items():
        new_key = parent_key + "." + k if parent_key else k
        if isinstance(v, dict):
            items.extend(nested_dict_to_dot_map_dict(v, new_key).items())
        else:
            items.append((new_key, v))

    return dict(items)


def list_of_dicts__to__dict_of_lists(lst, default_value=''):
    """
    ```
    x = [
        {'foo': 3, 'bar': 1},
        {'foo': 4, 'bar': 2},
        {'foo': 5},
        {'foo': 6, 'bar': 3},
        {'foo': 7},
    ]
    ppp.list_of_dicts__to__dict_of_lists(x, {'bar': 0})
    # Output:
    # {'foo': [3, 4, 5, 6, 7], 'bar': [1, 2, '', 3, '']}
    ```
    """

    if len(lst) == 0:
        return {}
    keys = reduce(set.union, [set(d.keys()) for d in lst])
    output_dict = collections.defaultdict(list)

    for d in lst:
        for k in keys:
            try:
                output_dict[k].append(d[k] if k in d.keys() else default_value)
            except KeyError:
                raise ValueError(f'no specified missing key, {k}')

    return output_dict
