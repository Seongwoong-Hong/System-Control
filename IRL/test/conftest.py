import os
import pytest
import gym_envs

from stable_baselines3.common.vec_env import DummyVecEnv
from scipy import io


@pytest.fixture()
def venv():
    env_type = "HPC"
    env_id = "{}_custom-v1".format(env_type)
    n_steps = 600
    sub = "sub01"
    pltqs = []
    for i in range(35):
        file = os.path.join("..", "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
        pltqs += [io.loadmat(file)['pltq']]
    env = DummyVecEnv([lambda: gym_envs.make(env_id, n_steps=n_steps, pltqs=pltqs) for _ in range(10)])
    return env


@pytest.fixture
def env():
    pltqs = []
    for i in range(35):
        file = os.path.join("..", "demos", "HPC", "sub01", "sub01i%d.mat" % (i + 1))
        pltqs += [io.loadmat(file)['pltq']]
    env = gym_envs.make("HPC_custom-v1", n_steps=600, pltqs=pltqs)
    return env


@pytest.fixture
def tenv():
    pltqs = []
    for i in range(35):
        file = os.path.join("..", "demos", "HPC", "sub01", "sub01i%d.mat" % (i + 1))
        pltqs += [io.loadmat(file)['pltq']]
    env = gym_envs.make("HPC_custom-v0", n_steps=600, pltqs=pltqs)
    return env
