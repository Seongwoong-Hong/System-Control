import os
import time
import pickle
import numpy as np

from matplotlib import pyplot as plt
from imitation.data.rollout import flatten_trajectories

from common.util import make_env


def run_traj(env, expert_dir):
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    for traj in expert_trajs:
        env.reset()
        tran = flatten_trajectories([traj])
        env.set_state(tran.obs[0, :env.model.nq], tran.obs[0, env.model.nq:env.model.nq+env.model.nv+1])
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
    env = make_env("IDP_custom-v2", use_vec_env=False)
    expert_dir = os.path.join("..", "demos", "IDP", "ppolearned.pkl")
    run_traj(env, expert_dir)


def test_ip():
    env = make_env("IP_custom-v1", use_vec_env=False, n_steps=600)
    expert_dir = os.path.join("..", "demos", "IP", "expert.pkl")
    run_traj(env, expert_dir)


def test_mujoco_envs():
    env = make_env("Ant-v2", use_vec_env=False)
    expert_dir = os.path.join("..", "demos", "mujoco_envs", "ant.pkl")
    run_traj(env, expert_dir)
