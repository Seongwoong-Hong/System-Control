import time
import numpy as np
import pickle

from scipy import signal
from imitation.data.rollout import flatten_trajectories

from algos.torch.OptCont import DiscreteLQRPolicy
from common.sb3.util import make_env


class IDPLQRPolicy(DiscreteLQRPolicy):
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
        self.Q = np.diag([0.7139, 0.5872182, 1.0639979, 0.9540204])
        self.R = np.diag([.061537065, .031358577])
        self.gear = 100


class IPLQRPolicy(DiscreteLQRPolicy):
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


def test_get_log_probs():
    env = make_env("IDP_custom-v2")
    with open(f"../../IRL/demos/IDP/uncropped/sub05_1.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    algo = IDPLQRPolicy(env, alpha=0.02)
    trans = flatten_trajectories(expt_trajs)
    algo.get_log_prob_from_act(trans.obs, trans.acts)
