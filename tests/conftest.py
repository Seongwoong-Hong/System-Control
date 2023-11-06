import os
import pytest
import pickle
import numpy as np
import torch as th
from scipy import signal

from algos.torch.MaxEntIRL import QuadraticRewardNet
from algos.torch.OptCont import *
from common.util import make_env
from common.wrappers import RewardWrapper


def feature_fn(x):
    return x


class IDPDiscLQRPolicy(DiscreteLQRPolicy):
    def _build_env(self) -> np.array:
        I1, I2 = self.env.envs[0].model.body_inertia[1:, 0]
        l1 = self.env.envs[0].model.body_pos[2, 2]
        lc1, lc2 = self.env.envs[0].model.body_ipos[1:, 2]
        m1, m2 = self.env.envs[0].model.body_mass[1:]
        g = 9.81
        M = np.array([[I1 + m1*lc1**2 + I2 + m2*l1**2 + 2*m2*l1*lc2 + m2*lc2**2, I2 + m2*l1*lc2 + m2*lc2**2],
                      [I2 + m2*l1*lc2 + m2*lc2**2, I2 + m2*lc2**2]])
        C = np.array([[m1*lc1*g + m2*l1*g + m2*g*lc2, m2*g*lc2],
                      [m2*g*lc2, m2*g*lc2]])
        self.A, self.B = np.zeros([4, 4]), np.zeros([4, 2])
        self.A[:2, 2:] = np.eye(2, 2)
        self.A[2:, :2] = np.linalg.inv(M) @ C
        self.B[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
        self.A, self.B, _, _, dt = signal.cont2discrete((self.A, self.B, np.array([1, 1, 1, 1]), 0), self.env.envs[0].dt)
        self.Q = self.env.envs[0].Q
        self.R = self.env.envs[0].R
        self.gear = 300


class IDPDiffLQRPolicy(DiffLQRPolicy):
    def _build_env(self):
        I1, I2 = self.env.envs[0].model.body_inertia[1:, 0]
        l1 = self.env.envs[0].model.body_pos[2, 2]
        lc1, lc2 = self.env.envs[0].model.body_ipos[1:, 2]
        m1, m2 = self.env.envs[0].model.body_mass[1:]
        g = 9.81
        M = np.array([[I1 + m1*lc1**2 + I2 + m2*l1**2 + 2*m2*l1*lc2 + m2*lc2**2, I2 + m2*l1*lc2 + m2*lc2**2],
                      [I2 + m2*l1*lc2 + m2*lc2**2, I2 + m2*lc2**2]])
        C = np.array([[m1*lc1*g + m2*l1*g + m2*g*lc2, m2*g*lc2],
                      [m2*g*lc2, m2*g*lc2]])
        self.A, self.B = np.zeros([4, 4]), np.zeros([4, 2])
        self.A[:2, 2:] = np.eye(2, 2)
        self.A[2:, :2] = np.linalg.inv(M) @ C
        self.B[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
        self.A, self.B, _, _, self.dt = signal.cont2discrete((self.A, self.B, np.eye(4), 0), self.env.envs[0].dt)
        self.max_t = 720
        self.Q = np.diag([0.7139, 0.5872182, 1.0639979, 0.9540204])
        self.R1 = np.diag([.061537065, .031358577])
        self.R2 = np.diag([.00, .00])
        # self.Q = np.diag([2, 1, 0.04, 0.04])
        # self.R = np.diag([0.005, 0.005])
        self.gear = 300


class IDPiterLQRPolicy(iterLQRPolicy):
    def _build_env(self) -> np.array:
        I1, I2 = self.env.envs[0].model.body_inertia[1:, 0]
        l1 = self.env.envs[0].model.body_pos[2, 2]
        lc1, lc2 = self.env.envs[0].model.body_ipos[1:, 2]
        m1, m2 = self.env.envs[0].model.body_mass[1:]
        g = 9.81
        M = np.array([[I1 + m1*lc1**2 + I2 + m2*l1**2 + 2*m2*l1*lc2 + m2*lc2**2, I2 + m2*l1*lc2 + m2*lc2**2],
                      [I2 + m2*l1*lc2 + m2*lc2**2, I2 + m2*lc2**2]])
        C = np.array([[m1*lc1*g + m2*l1*g + m2*g*lc2, m2*g*lc2],
                      [m2*g*lc2, m2*g*lc2]])
        A, B = np.zeros([4, 4]), np.zeros([4, 2])
        A[:2, 2:] = np.eye(2, 2)
        A[2:, :2] = np.linalg.inv(M) @ C
        B[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
        self.A, self.B, _, _, dt = signal.cont2discrete((A, B, np.array([1, 1, 1, 1]), 0), self.env.envs[0].dt)
        self.Q = th.from_numpy(np.diag([0.7139, 0.5872182, 1.0639979, 0.9540204])).float()
        self.R = th.from_numpy(np.diag([.061537065, .031358577])).float()
        self.f_x = lambda x, u: th.from_numpy(self.A).float()
        self.f_u = lambda x, u: th.from_numpy(self.B).float()
        self.l = lambda x, u: th.sum((x @ self.Q) * x, dim=-1) + th.sum((u @ self.R) * u, dim=-1)
        self.lf_x = lambda x: x @ self.Q
        self.lf_xx = lambda x: self.Q
        self.max_t = 360
        self.gear = 300


proj_path = os.path.abspath("../..")
with open(f"{proj_path}/IRL/demos/HPC/full/sub01_1.pkl", "rb") as f:
    expt_trajs = pickle.load(f)
init_states = []
pltqs = []
for traj in expt_trajs:
    init_states.append(traj.obs[0])
    pltqs.append(traj.pltq)


@pytest.fixture
def idpilqrpolicy():
    policy = IDPiterLQRPolicy
    return policy


@pytest.fixture
def idpdiffpolicy():
    policy = IDPDiffLQRPolicy
    return policy


@pytest.fixture
def idpdisclqrpolicy():
    policy = IDPDiscLQRPolicy
    return policy


@pytest.fixture
def hpc_env():
    env = make_env("HPC_custom-v2", init_states=init_states, pltqs=pltqs)
    return env


@pytest.fixture
def hpc_with_rwrap_env():
    reward_fn = QuadraticRewardNet(inp=6, arch=[8, 8], feature_fn=feature_fn, use_action_as_inp=True, device='cpu')
    env = make_env("HPC_custom-v2", init_states=init_states, pltqs=pltqs, wrapper=RewardWrapper, wrapper_kwargs={'rwfn': reward_fn})
    return env


@pytest.fixture
def irl_path():
    return os.path.abspath(os.path.join("..", "..", "IRL"))