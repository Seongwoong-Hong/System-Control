import time
import gym_envs
import numpy as np

from algos.torch.OptCont import LQRPolicy


class IDPpolicy(LQRPolicy):
    def _build_env(self):
        m1, m2 = self.env.m1, self.env.m2
        h1, h2 = self.env.com1, self.env.com2
        I1, I2 = self.env.I1, self.env.I2
        self.gear = self.env.action_coeff
        g = 9.81
        self.Q = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0.1, 0],
                           [0, 0, 0, 0.1]])
        self.R = 1e-5*np.array([[1, 0],
                                [0, 1]])
        self.A = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [m1*g*h1/I1, 0, 0, 0],
                           [0, m2*g*h2/I2, 0, 0]])
        self.B = np.array([[0, 0],
                           [0, 0],
                           [1/I1, -1/I1],
                           [0, 1/I2]])
        return self.A, self.B, self.Q, self.R


def test_cus_idp():
    env = gym_envs.make("IDP_classic-v0")
    done = False
    env.reset()
    env.render()
    while not done:
        obs, rew, done, _ = env.step(env.action_space.sample())
        env.render()
        time.sleep(0.005)
    env.close()
    print(env.action_space.sample())


def test_environment_action():
    env = gym_envs.make("IDP_classic-v0")
    algo = IDPpolicy(env)
    done = False
    obs = env.reset()
    env.render()
    while not done:
        act, _ = algo.predict(obs, deterministic=True)
        obs, rew, done, _ = env.step(act)
        env.render()
        time.sleep(env.dt)
    env.close()
