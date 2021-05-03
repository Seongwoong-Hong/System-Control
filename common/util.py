import os
import os.path as p
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
                file = p.join("demos", env_type, sub, sub + "i%d.mat" % (i + 1))
                if not p.isfile(file):
                    break
                pltqs += [io.loadmat(file)['pltq']]
                i += 1
            kwargs['pltqs'] = pltqs
        if use_vec_env:
            env = DummyVecEnv([lambda: gym_envs.make(env_name, **kwargs) for _ in range(num_envs)])
        else:
            env = gym_envs.make(env_name, **kwargs)
    elif env_type == "IDP":
        kwargs.pop('pltqs', None)
        if use_vec_env:
            env = DummyVecEnv([lambda: gym_envs.make(env_name, **kwargs) for _ in range(num_envs)])
        else:
            env = gym_envs.make(env_name, **kwargs)
    else:
        raise NameError('Unknown environment type')
    return env


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


def create_path(dirname=None, filename=None):
    assert dirname is not None or filename is not None, "Please enter directory or file name"
    assert dirname is None or filename is None, "One of input file name or directory is none"
    if filename is not None:
        dirname = os.path.abspath(os.path.join(filename, os.pardir))
    paths = []
    while not os.path.isdir(dirname):
        paths.append(dirname)
        dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    for path in reversed(paths):
        os.mkdir(path)
