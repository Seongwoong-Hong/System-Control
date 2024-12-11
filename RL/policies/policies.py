from algos.torch.OptCont import *

import torch as th
import numpy as np
from scipy import signal


class HeadTrackLinearFeedback(LinearFeedbackPolicy):
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        obs = observation[0, :4]
        headx = -(self.model.geom_pos[1, -1] * th.sin(obs[0]) + self.model.geom_pos[2, -1] * th.sin(obs[0] + obs[1]))
        headv = -(self.model.geom_pos[1, -1] * th.cos(obs[0]) + self.model.geom_pos[2, -1] * th.cos(obs[0] + obs[1]))
        tg = observation[0, 4:]
        err = (tg - np.array([headx, headv]))[:, None]
        if self.K is None:
            raise Exception("피드백 게인이 정의되지 않았습니다")
        noise = 0
        if not deterministic:
            noise = self.noise_lv * np.random.randn(*self.K.shape)
        return 1 / self.gear * (self.K @ err)

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
        self.A, self.B, _, _, self.dt = signal.cont2discrete((A, B, np.eye(4), 0), self.env.envs[0].dt)
        # self.Q = th.from_numpy(np.diag([0.7139, 0.5872182, 1.0639979, 0.9540204])).float()
        # self.R = th.from_numpy(np.diag([.061537065, .031358577])).float()
        self.Q = th.diag(th.tensor([2., 1., 0.04, 0.04]))
        self.R = th.diag(th.tensor([0.005, 0.005]))
        self.q = th.tensor([0.1, 0.1, 0.0, 0.0])
        self.r = th.tensor([0.0, 0.0])
        self.f_x = lambda x, u: th.from_numpy(self.A).float()
        self.f_u = lambda x, u: th.from_numpy(self.B).float()
        self.l = lambda x, u: th.sum(((x - self.q) @ self.Q) * (x - self.q), dim=-1) + th.sum(((u - self.r) @ self.R) * (u - self.r), dim=-1)
        self.lf = lambda x: th.sum(((x - self.q) @ self.Q) * (x - self.q), dim=-1)


class IDPLQRPolicy(LQRPolicy):
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
        # self.A, self.B, _, _, self.dt = signal.cont2discrete((self.A, self.B, np.eye(4), 0), self.env.envs[0].dt)
        self.Q = np.diag([1.08932670e-00, 1.00035577e-01, 9.92401693e-02, 9.97238859e-02,])
        self.R = np.diag([1.00052541e-05, 1.07954182e-05,])


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
        self.Q = np.diag([1.08932670e-00, 1.00035577e-01, 9.92401693e-02, 9.97238859e-02,])
        # self.R1 = np.diag([1.00052541e-03, 1.07954182e-04,])
        self.R1 = np.diag([0, 0])
        self.R2 = np.diag([7.27747447e-06, 3.29719842e-05])
        # self.Q = np.diag([2, 1, 0.04, 0.04])
        # self.R = np.diag([0.005, 0.005])


class IPLQRPolicy(LQRPolicy):
    def _build_env(self):
        I = self.env.envs[0].model.body_inertia[1, 0]
        lc = self.env.envs[0].model.body_ipos[1, 2]
        m = self.env.envs[0].model.body_mass[1]
        g = 9.81
        self.A = np.array([[0, 1], [(m*g*lc)/(I + m*lc**2), 0]])
        self.B = np.array([[0], [1/(I + m*lc**2)]])
        self.Q = np.diag([2.1465e-06, 2.3304e+00])
        self.R = np.diag([2.3219e+00 / 90000])
