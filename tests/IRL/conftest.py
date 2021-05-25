import os
import pytest
import gym_envs

from stable_baselines3.common.vec_env import DummyVecEnv
from scipy import io


@pytest.fixture
def pltqs():
    pltqs = []
    sub = "sub01"
    for i in range(35):
        file = os.path.join("..", "..", "IRL", "demos", "HPC", sub, sub + "i%d.mat" % (i + 1))
        pltqs += [io.loadmat(file)['pltq']]
    return pltqs


@pytest.fixture
def venv(pltqs):
    env_type = "HPC"
    env_id = "{}_custom-v1".format(env_type)
    n_steps = 600
    env = DummyVecEnv([lambda: gym_envs.make(env_id, n_steps=n_steps, pltqs=pltqs) for _ in range(10)])
    return env


@pytest.fixture
def env(pltqs):
    env = gym_envs.make("HPC_custom-v1", n_steps=600, pltqs=pltqs)
    return env


@pytest.fixture
def tenv(pltqs):
    env = gym_envs.make("HPC_custom-v0", n_steps=600, pltqs=pltqs)
    return env
