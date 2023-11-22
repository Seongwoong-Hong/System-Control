import copy
import pickle
import time
import numpy as np
from common.util import make_env


def test_hpc_env_reset(hpc_env):
    hpc_env.reset()
    done = False
    while not done:
        act = hpc_env.action_space.sample()
        ob, rew, done, info = hpc_env.step(act)


def test_init():
    env = make_env("IP_custom-v2", subpath="../../IRL/demos/HPC/sub01/sub01")
    # env.render()
    env.reset()
    # env.set_state(np.array([0.0, 0.3]), np.array([0.0, 0.0]))
    done = False
    for _ in range(1000):
        env.set_state(np.array([0.3]), np.array([0]))
        _, _, _, _ = env.step(np.array([.5]))
        # env.render()
        print(env.current_obs)
        time.sleep(env.dt)


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
