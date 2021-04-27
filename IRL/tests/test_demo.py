import os
import time
import pickle
import numpy as np

from matplotlib import pyplot as plt
from imitation.data.rollout import flatten_trajectories


def run_traj(env, expert_dir):
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    for traj in expert_trajs:
        env.reset()
        tran = flatten_trajectories([traj])
        env.set_state(tran.obs[0, :2], tran.obs[0, 2:4])
        if hasattr(env, "pltq"):
            env.pltq = traj.obs[:, 4:]
        for t in range(len(tran)):
            act = tran.acts[t]
            obs, _, _, _ = env.step(act)
            env.render()
            time.sleep(env.dt)
    env.close()


def test_hpc(env):
    expert_dir = os.path.join("..", "demos", "HPC", "lqrTest.pkl")
    run_traj(env, expert_dir)


def test_hpcdiv(env):
    expert_dir = os.path.join("..", "demos", "HPC", "lqrDivTest.pkl")
    run_traj(env, expert_dir)


def test_idp():
    import gym_envs
    env = gym_envs.make("IDP_custom-v1", n_steps=600)
    expert_dir = os.path.join("..", "demos", "IDP", "expert.pkl")
    run_traj(env, expert_dir)