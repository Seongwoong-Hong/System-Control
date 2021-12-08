import time
import os
import pickle
import numpy as np
import pytest

from algos.torch.ppo import PPO
from common.util import make_env
from common.rollouts import generate_trajectories_without_shuffle

from imitation.data.rollout import make_sample_until, flatten_trajectories
from stable_baselines3.common.vec_env import DummyVecEnv

env_id = "DiscretePendulum"
env_op = 0.03
agent_path = os.path.join("..", "..", "RL", env_id, "tmp", "log", f"{env_id}_{env_op}", "softqiter", "policies_1",
                          "agent")


@pytest.fixture
def agent():
    with open(agent_path + ".pkl", "rb") as f:
        ag = pickle.load(f)
    return ag


def test_rollout_fn(agent):
    subpath = os.path.join("..", "..", "IRL", "demos", "HPC", "sub01", "sub01")
    venv = make_env(f"{env_id}-v2", num_envs=1, wrapper="None", subpath=subpath, h=env_op)
    sample_until = make_sample_until(n_timesteps=None, n_episodes=35)
    trajectories = generate_trajectories_without_shuffle(agent, venv, sample_until, deterministic_policy=False)
    trans = flatten_trajectories(trajectories)


def test_custom_rollout():
    agent_path = os.path.join("..", "..", "RL", "HPC", "tmp", "log", "HPC_custom", "ppo", "policies_4", "agent")
    subpath = os.path.join("..", "..", "IRL", "demos", "HPC", "sub01", "sub01")
    venv = make_env("HPC_custom-v0", num_envs=5, subpath=subpath, wrapper="ActionWrapper")
    agent = PPO.load(agent_path)
    agent.set_env(venv)
    ob = venv.reset()
    obs = ob[..., np.newaxis]
    dones = np.array([False for _ in range(5)])
    while not dones.any():
        acts, _ = agent.predict(ob, deterministic=False)
        ob, _, dones, infos = venv.step(acts)
        obs = np.concatenate([obs, ob[..., np.newaxis]], axis=2)

    print('end')
