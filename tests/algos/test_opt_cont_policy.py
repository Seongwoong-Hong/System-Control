import time
import numpy as np

from algos.torch.OptCont import LQRPolicy

from common.util import make_env


class IDPLQRPolicy(LQRPolicy):
    def _build_env(self) -> np.array:
        I1, I2 = 0.878121, 1.047289
        l1 = 0.7970
        lc1, lc2 = 0.5084, 0.2814
        m1 ,m2 = 17.2955, 34.5085
        g = 9.81
        M = np.array([[I1 + m1*lc1**2 + I2 + m2*l1**2 + 2*m2*l1*lc2 + m2*lc2**2, I2 + m2*l1*lc2 + m2*lc2**2],
                      [I2 + m2*l1*lc2 + m2*lc2**2, I2 + m2*lc2**2]])
        C = np.array([[m1*lc1*g + m2*l1*g + m2*g*lc2, m2*g*lc2],
                      [m2*g*lc2, m2*g*lc2]])
        self.A, self.B = np.zeros([4, 4]), np.zeros([4, 2])
        self.A[:2, 2:] = np.eye(2, 2)
        self.A[2:, :2] = np.linalg.inv(M) @ C
        self.B[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
        self.Q = np.diag([3.5139, 0.7872182, 0.14639979, 0.07540204])
        self.R = np.diag([0.02537065/1600, 0.01358577/900])
        self.gear = 1


class IPLQRPolicy(LQRPolicy):
    def _build_env(self) -> np.array:
        g = 9.81
        m = 1.
        l = 1.
        lc = l / 2
        I = m * l ** 2 / 3
        self.A, self.B = np.zeros([2, 2]), np.zeros([2, 1])
        self.A[0, 1] = 1
        self.A[1, 0] = m * g * lc / I
        self.B[1, 0] = 1 / I
        self.Q = np.diag([2.5139, 0.2872182])
        self.R = np.diag([0.01537065/2500])
        self.gear = 100


def test_lqr_policy():
    # env = make_env("DiscretizedPendulum-v2", N=[19, 19], NT=[11])
    env = make_env("DiscretizedHuman-v2", N=[19, 19, 19, 19], NT=[11, 11])
    policy = IDPLQRPolicy(env)
    for _ in range(10):
        ob = env.reset()
        env.render()
        time.sleep(env.dt)
        done = False
        while not done:
            act, _ = policy.predict(ob, deterministic=True)
            ob, _, done, _ = env.step(act)
            env.render()
            time.sleep(env.dt)
    env.close()
