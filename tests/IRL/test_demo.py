import os
import pytest
import time
import pickle
import numpy as np

from scipy import io
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


@pytest.fixture
def pltqs(demo_dir):
    pltqs = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        file = os.path.join(demo_dir, "HPC", "sub01", f"sub01i{i + 1}.mat")
        pltqs += [io.loadmat(file)['pltq']]
    return pltqs


def test_hpc(demo_dir, pltqs):
    env = make_env("HPC_custom-v0", pltqs=pltqs)
    expert_dir = os.path.join(demo_dir, "HPC", "sub01.pkl")
    run_traj(env, expert_dir)


def test_hpcdiv(demo_dir, pltqs):
    env = make_env("HPC_custom-v0", pltqs=pltqs)
    expert_dir = os.path.join(demo_dir, "HPC", "lqrDivTest.pkl")
    run_traj(env, expert_dir)


def test_idp(demo_dir):
    env = make_env("IDP_custom-v0", use_vec_env=False)
    expert_dir = os.path.join(demo_dir, "IDP", "lqr_known.pkl")
    run_traj(env, expert_dir)


def test_ip():
    env = make_env("IP_custom-v1", use_vec_env=False, n_steps=600)
    expert_dir = os.path.join("..", "demos", "IP", "expert.pkl")
    run_traj(env, expert_dir)


def test_mujoco_envs():
    env = make_env("Ant-v2", use_vec_env=False)
    expert_dir = os.path.join("..", "demos", "mujoco_envs", "ant.pkl")
    run_traj(env, expert_dir)
