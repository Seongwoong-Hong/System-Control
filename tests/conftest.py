import os
import pytest
import pickle
import numpy as np
import torch as th
from scipy import signal, io

from algos.torch.MaxEntIRL import QuadraticRewardNet
from algos.torch.OptCont import *
from common.util import make_env
from common.wrappers import RewardWrapper


def feature_fn(x):
    return x


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


@pytest.fixture
def idpilqrpolicy():
    policy = IDPiterLQRPolicy
    return policy


@pytest.fixture
def idpdiffpolicy():
    policy = IDPDiffLQRPolicy
    return policy


@pytest.fixture
def irl_path():
    return os.path.abspath(os.path.join("..", "..", "IRL"))


@pytest.fixture
def rl_path():
    return os.path.abspath(os.path.join("..", "..", "RL"))


@pytest.fixture
def proj_path():
    return os.path.abspath(os.path.join("..", ".."))


@pytest.fixture
def IPhumanData(proj_path):
    subpath = os.path.join(proj_path, "demos", "IP", "sub04", "sub04")
    return io.loadmat(subpath + "i11.mat")


@pytest.fixture
def IPhumanStates(IPhumanData):
    states = [None for _ in range(35)]
    states[11 - 1] = IPhumanData['state']
    return states


@pytest.fixture
def IPbsp(IPhumanData):
    return IPhumanData['bsp']


@pytest.fixture
def ip_env(IPbsp, IPhumanStates):
    return make_env(f"IP_MimicHuman-v2", bsp=IPbsp, humanStates=IPhumanStates)


@pytest.fixture
def ip_env_norm(proj_path, IPbsp, IPhumanStates):
    # norm_path = os.path.join(proj_path, "RL", "scripts", "tmp", "log", "PseudoIP_norm", "ppo_DeepMimic_PD_ptb3", "policies_3", "normalization_2.pkl")
    norm_path = None
    if norm_path is None:
        norm_path = True
    return make_env(f"IP_MimicHuman-v2", num_envs=8, bsp=IPbsp, humanStates=IPhumanStates, use_norm=norm_path)


@pytest.fixture
def ip_env2_norm(IPbsp, IPhumanStates):
    return make_env(f"IP_MinEffort-v2", num_envs=1, bsp=IPbsp, humanStates=IPhumanStates, use_norm=True, w=0.9)


@pytest.fixture
def ip_env_vec(IPbsp, IPhumanStates):
    return make_env(f"IP_MimicHuman-v2", num_envs=8, bsp=IPbsp, humanStates=IPhumanStates)


@pytest.fixture
def ip_env2_vec(IPbsp, IPhumanStates):
    return make_env(f"IP_MinEffort-v2", num_envs=8, bsp=IPbsp, humanStates=IPhumanStates, w=0.9)


@pytest.fixture
def idp_env(proj_path):
    subpath = os.path.join(proj_path, "demos", "IDP", "sub04", "sub04")
    states = [None for _ in range(35)]
    for i in range(31, 36):
        humanData = io.loadmat(subpath + f"i{i}.mat")
        states[i - 1] = humanData['state']
        bsp = humanData['bsp']
    return make_env("IDP_MimicHuman-v2", num_envs=8, bsp=bsp, humanStates=states, ankle_max=100)


@pytest.fixture
def idp_env2(proj_path):
    subpath = os.path.join(proj_path, "demos", "IDP", "sub04", "sub04")
    states = [None for _ in range(35)]
    for i in range(31, 36):
        humanData = io.loadmat(subpath + f"i{i}.mat")
        states[i - 1] = humanData['state']
        bsp = humanData['bsp']
    return make_env("IDP_MinEffort-v2", num_envs=1, bsp=bsp, humanStates=states, ankle_max=100, w=0.9)