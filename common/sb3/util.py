import argparse
import json
import os
import pickle
import warnings
import os.path as p
from pathlib import Path
import torch

import gym_envs
import gym
from io import BytesIO

from scipy import io
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from common.path_config import MAIN_DIR
from common.util import str2bool
from common.wrappers import ActionWrapper
from algos.torch.sb3.ppo import PPO


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


def load_result(
        env_type,
        log_dir: Path,
        algo_num: int,
        env_id=None,
        env_kwargs=None,
        device='cpu',
):
    if env_id is not None:
        env_id = f"{env_type}_{env_id}"
    else:
        env_id = env_type

    if not log_dir.is_dir():
        raise Exception("Model directory doesn't exist")

    with open(f"{str(log_dir)}/config.json", 'r') as f:
        config = json.load(f)

    if "isPseudo" in config:
        if str2bool(config["isPseudo"]):
            env_type = "Pseudo" + env_type

    if str2bool(config['use_norm']):
        norm_pkl_path = str(log_dir / f"normalization_{algo_num}.pkl")
    else:
        norm_pkl_path = False

    if env_kwargs is not None:
        for k, v in env_kwargs.items():
            config['env_kwargs'][k] = v

    loaded_result = {}
    if "IDP" in env_type:
        subj = config['subj']
        subpath = MAIN_DIR / "demos" / env_type / subj / subj
        if env_type == "IDPPD":
            subpath = MAIN_DIR / "demos" / "IDP" / subj / subj

        states = [None for _ in range(35)]
        torques = [None for _ in range(35)]

        except_trials = env_kwargs.pop("except_trials", [])
        trials = config['env_kwargs'].pop("trials", [])

        for rm_trial in except_trials:
            if rm_trial in trials:
                trials.remove(rm_trial)

        if len(trials) == 0:
            raise Exception("No trials found")

        for trial in trials:
            humanData = io.loadmat(str(subpath) + f"i{trial}.mat")
            bsp = humanData['bsp']
            states[trial - 1] = humanData['state']
            torques[trial - 1] = humanData['tq']

        config['env_kwargs']['bsp'] = bsp
        config['env_kwargs']['humanStates'] = states
        comp_states = [state for state in states if state is not None]
        comp_torques = [torque for torque in torques if torque is not None]
        loaded_result = {"states": comp_states, "bsp": bsp, "torques": comp_torques, "save_dir": str(log_dir), "config": config}

    env_kwargs = config.pop("env_kwargs", {})
    env = make_env(f"{env_id}-v0", num_envs=1, use_norm=norm_pkl_path, **env_kwargs)

    agent = PPO.load(str(log_dir / f"agent_{algo_num}"), device=device)

    return env, agent, loaded_result


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
