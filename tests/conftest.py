import os
import isaacgym
import pytest
from scipy import signal, io

from algos.torch.OptCont import *
from common.sb3.util import make_env


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
def proj_path():
    return os.path.abspath(os.path.join(__file__, "..", ".."))


@pytest.fixture
def irl_path(proj_path):
    return os.path.join(proj_path, "IRL")


@pytest.fixture
def rl_path(proj_path):
    return os.path.join(proj_path, "RL")


# ==============================
# Single pendulum variables
# ==============================

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
    return make_env(f"IP_MimicHuman-v2", num_envs=8, bsp=IPbsp, humanStates=IPhumanStates, use_norm=True)


@pytest.fixture
def ip_env2_norm(IPbsp, IPhumanStates):
    return make_env(f"IP_MinEffort-v2", num_envs=1, bsp=IPbsp, humanStates=IPhumanStates, use_norm=True, w=0.9)


@pytest.fixture
def ip_env_vec(IPbsp, IPhumanStates):
    return make_env(f"IP_MimicHuman-v2", num_envs=8, bsp=IPbsp, humanStates=IPhumanStates)


@pytest.fixture
def ip_env2_vec(IPbsp, IPhumanStates):
    return make_env(f"IP_MinEffort-v2", num_envs=8, bsp=IPbsp, humanStates=IPhumanStates, w=0.9)


# ==============================
# Double pendulum variables
# ==============================

@pytest.fixture
def IDPhumanData(proj_path):
    subpath = os.path.join(proj_path, "demos", "IDP", "sub04", "sub04")
    return io.loadmat(subpath + "i11.mat")


@pytest.fixture
def IDPhumanStates(IDPhumanData):
    states = [None for _ in range(35)]
    states[11 - 1] = IDPhumanData['state']
    return states


@pytest.fixture
def IDPbsp(IDPhumanData):
    return IDPhumanData['bsp']


@pytest.fixture
def idp_det_env(IDPbsp, IDPhumanStates):
    return make_env("IDPPD_MinEffort-v0", bsp=IDPbsp, humanStates=IDPhumanStates)


@pytest.fixture
def idp_env(IDPbsp, IDPhumanStates):
    return make_env("IDPPD_MinEffort-v2", bsp=IDPbsp, humanStates=IDPhumanStates)


@pytest.fixture
def idp_env_vec(IDPbsp, IDPhumanStates):
    return make_env(
        "IDP_MinMetCost-v2",
        num_envs=8,
        bsp=IDPbsp,
        humanStates=IDPhumanStates,
        ankle_limit='soft',
        ankle_torque_max=120,
        vel_ratio=0.01,
        stiffness = [300, 50],
        damping = [30, 20],
    )
