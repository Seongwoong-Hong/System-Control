from algo.torch.OptCont import LQRPolicy
from algo.torch.ppo import PPO
from algo.torch.sac import SAC
import numpy as np


class IPPolicy(LQRPolicy):
    def _build_env(self):
        m, g, h, I = 5.0, 9.81, 0.5, 1.667
        self.Q = np.array([[1, 0], [0, 0.5]])
        self.R = 0.0001*np.array([[1]])
        self.A = np.array([[0, m*g*h/I], [1, 0]])
        self.B = np.array([[1/I], [0]])
        return self.A, self.B, self.Q, self.R


class IDPPolicy(LQRPolicy):
    def _build_env(self):
        m1, m2, h1, h2, I1, I2, g = 5.0, 5.0, 0.5, 0.5, 1.667, 1.667, 9.81
        # m1, m2, h1, h2, I1, I2, g = 22.0892, 46.5108, 0.50982, 0.30624, 7.937, 11.996, 9.81
        self.Q = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        self.R = 1e-6*np.array([[1, 0],
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


class HPCPolicy(LQRPolicy):
    def _build_env(self):
        m1, m2, h1, h2, I1, I2, g = 22.0892, 46.5108, 0.50982, 0.30624, 7.937, 11.996, 9.81
        self.Q = np.array([[0.15, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        self.R = 1e-5*np.array([[1, 0],
                                [0, 1]])
        self.A = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [m1*g*h1/I1, 0, 0, 0],
                           [0, m2*g*h2/I2, 0, 0]])
        self.B = np.array([[0, 0], [0, 0],
                           [1/I1, -1/I1], [0, 1/I2]])
        return self.A, self.B, self.Q, self.R


def def_policy(algo_type, env, device='cpu', log_dir=None, verbose=0, action_space=None, observation_space=None):
    if algo_type == "IP":
        return IPPolicy(env)
    elif algo_type == "IDP":
        return IDPPolicy(env)
    elif algo_type == "HPC":
        return HPCPolicy(env, action_space=action_space, observation_space=observation_space)
    elif algo_type == "ppo":
        from algo.torch.ppo import MlpPolicy
        return PPO(MlpPolicy,
                   env=env,
                   n_steps=1024,
                   batch_size=512,
                   gamma=0.975,
                   gae_lambda=0.95,
                   ent_coef=0.015,
                   ent_schedule=1.0,
                   clip_range=0.175,
                   verbose=verbose,
                   device=device,
                   tensorboard_log=log_dir,
                   policy_kwargs={'log_std_range': [-5, 5],
                                  'net_arch': [{'pi': [128, 128], 'vf': [128, 128]}]},
                   )

    elif algo_type == "sac":
        from algo.torch.sac import MlpPolicy
        return SAC(MlpPolicy,
                   env=env,
                   batch_size=256,
                   learning_starts=256,
                   train_freq=128,
                   gamma=0.99,
                   ent_coef='auto_0.1',
                   verbose=verbose,
                   device=device,
                   tensorboard_log=log_dir,
                   policy_kwargs={'net_arch': {'pi': [64, 64], 'qf': [64, 64]}},
                   )
    else:
        raise NameError("Not implemented policy name")
