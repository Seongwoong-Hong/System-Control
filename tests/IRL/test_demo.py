import os
import pytest
import time
import pickle
import numpy as np

from matplotlib import pyplot as plt
from imitation.data.rollout import flatten_trajectories

from common.util import make_env


def run_traj(env, expert_dir):
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    costs = []
    for traj in expert_trajs:
        rews = 0
        env.reset()
        tran = flatten_trajectories([traj])
        env.set_state(tran.obs[0, :env.model.nq], tran.obs[0, env.model.nq:env.model.nq+env.model.nv])
        if hasattr(env, "pltq"):
            env.pltq = traj.obs[:-1, 4:]
        for t in range(len(tran)):
            act = tran.acts[t]
            obs, rew, _, _ = env.step(act)
            rews += rew
            env.render()
            time.sleep(env.dt)
        costs.append(-rews)
    print(costs)
    env.close()


@pytest.fixture
def demo_dir():
    return os.path.abspath(os.path.join("..", "..", "IRL", "demos"))


def test_hpc(env, demo_dir):
    expert_dir = os.path.join(demo_dir, "HPC", "sub01_1&2.pkl")
    run_traj(env, expert_dir)


def test_hpcdiv(env, demo_dir):
    expert_dir = os.path.join(demo_dir, "HPC", "lqrDivTest.pkl")
    run_traj(env, expert_dir)


def test_idp(demo_dir):
    env = make_env("IDP_custom-v0", use_vec_env=False)
    expert_dir = os.path.join(demo_dir, "IDP", "lqr1.pkl")
    run_traj(env, expert_dir)


def test_ip():
    env = make_env("IP_custom-v1", use_vec_env=False, n_steps=600)
    expert_dir = os.path.join("..", "demos", "IP", "expert.pkl")
    run_traj(env, expert_dir)


def test_mujoco_envs():
    env = make_env("Ant-v2", use_vec_env=False)
    expert_dir = os.path.join("..", "demos", "mujoco_envs", "ant.pkl")
    run_traj(env, expert_dir)
