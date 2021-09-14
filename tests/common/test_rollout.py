import time
import os
import numpy as np

from algos.torch.ppo import PPO
from common.util import make_env
from common.rollouts import generate_trajectories_without_shuffle

from imitation.data.rollout import make_sample_until, flatten_trajectories
from stable_baselines3.common.vec_env import DummyVecEnv


def test_vec_rollout():
    def rwfn(obs, acts):
        return (np.square(obs[:, :2]) + 0.1*np.square(obs[:, 2:4]) + 1e-6*np.square(acts)).sum()
    agent_path = os.path.join("..", "..", "RL", "HPC", "tmp", "log", "HPC_custom", "ppo", "policies_4", "agent")
    subpath = os.path.join("..", "..", "IRL", "demos", "HPC", "sub01", "sub01")
    start = time.time()
    venv = make_env("HPC_custom-v0", use_vec_env=True, num_envs=1, wrapper="ActionWrapper", subpath=subpath)
    agent = PPO.load(agent_path)
    sample_until = make_sample_until(n_timesteps=None, n_episodes=35)
    trajectories = generate_trajectories_without_shuffle(agent, venv, sample_until, deterministic_policy=False)
    trans = flatten_trajectories(trajectories)
    reward = rwfn(trans.obs, trans.acts)
    print(reward)


def test_custom_rollout():
    agent_path = os.path.join("..", "..", "RL", "HPC", "tmp", "log", "HPC_custom", "ppo", "policies_4", "agent")
    subpath = os.path.join("..", "..", "IRL", "demos", "HPC", "sub01", "sub01")
    venv = make_env("HPC_custom-v0", use_vec_env=True, num_envs=5, subpath=subpath, wrapper="ActionWrapper")
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
