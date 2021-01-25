from algo.torch.OptCont import LQRPolicy
import numpy as np

class IPPolicy(LQRPolicy):
    def _build_env(self):
        m, g, h, I = 5.0, 9.81, 0.5, 1.667
        self.Q = np.array([[1, 0], [0, 1]])
        self.R = 0.001*np.array([[1]])
        self.A = np.array([[0, m*g*h/I], [1, 0]])
        self.B = np.array([[1/I], [0]])
        return self.A, self.B, self.Q, self.R

class IDPPolicy(LQRPolicy):
    def _build_env(self):
        m1, m2, h1, h2, I1, I2, g = 5.0, 5.0, 0.5, 0.5, 1.667, 1.667, 9.81
        self.Q = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        self.R = 0.0001*np.array([[1, 0],
                             [0, 1]])
        self.A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [m1*g*h1/I1, 0, 0, 0],
                      [0, m2*g*h2/I2, 0, 0]])
        self.B = np.array([[0, 0], [0, 0],
                      [1/I1, -1/I1], [0, 1/I2]])
        return self.A, self.B, self.Q, self.R

def def_policy(env_type, env):
    if env_type == "IP":
        return IPPolicy(env)
    elif env_type == "IDP":
        return IDPPolicy(env)
    else:
        raise NameError("Not defined policy name")