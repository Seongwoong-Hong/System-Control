import copy
import pickle
import time
import numpy as np
from common.util import make_env


def test_env_init(ip_env):
    env = ip_env
    env.close()


def test_pickle(ip_env_norm):
    env = ip_env_norm
    pickled_env = pickle.dumps(env)
    unpickled_env = pickle.loads(pickled_env)
    print(unpickled_env)


def test_deepcopy(ip_env_norm):
    env = ip_env_norm
    # getattr(ip_env_norm, "abc")
    copied = copy.deepcopy(env)
    copied2 = copy.deepcopy(copied)
    print(copied.class_attributes)


def test_vecenv_state(ip_env_vec):
    env = ip_env_vec
    obs = env.reset()
    print(obs)
