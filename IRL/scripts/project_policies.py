import numpy as np
import torch as th

from algos.torch.OptCont import LQRPolicy
from algos.torch.ppo import PPO
from algos.torch.sac import SAC


class IPPolicy(LQRPolicy):
    def _build_env(self):
        m, g, h, I = 5.0, 9.81, 0.5, 1.667
        self.gear = self.env.model.actuator_gear[0, 0]
        self.Q = np.array([[1, 0], [0, 0.5]])
        self.R = 0.0001*np.array([[1]])
        self.A = np.array([[0, m*g*h/I], [1, 0]])
        self.B = np.array([[1/I], [0]])
        return self.A, self.B, self.Q, self.R


class IDPPolicy(LQRPolicy):
    def _build_env(self):
        _, m1, m2 = self.env.model.body_mass
        _, h1, h2 = self.env.model.body_ipos[:, 2]
        _, I1, I2 = self.env.model.body_inertia[:, 1]
        self.gear = self.env.model.actuator_gear[0, 0]
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


class HPCPolicy(IDPPolicy):
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        noise = 0
        if not deterministic:
            noise = self.noise_lv * np.random.randn(*self.K.shape)

        return -1/self.gear * ((self.K + noise) @ observation[:, :4].T).reshape(1, -1)


class HPCDivPolicy(IDPPolicy):
    def __init__(self, env, noise_lv: float = 0.1,
                 observation_space=None,
                 action_space=None,
                 ):
        super().__init__(env, noise_lv, observation_space, action_space)
        self.Q = np.array([[0.5, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0.05, 0],
                           [0, 0, 0, 0.05]])
        self.K2 = self._get_gains()

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        if self.env.order <= 20:
            K = self.K
        else:
            K = self.K2
        if not deterministic:
            K += self.noise_lv * np.random.randn(*K.shape)
        return -1/self.gear * (K @ observation[:, :4].T).reshape(1, -1)


def def_policy(algo_type, env, device='cpu', log_dir=None, verbose=0, **kwargs):
    if algo_type == "IP":
        return IPPolicy(env, **kwargs)
    elif algo_type == "IDP":
        return IDPPolicy(env, **kwargs)
    elif algo_type == "HPC":
        return HPCPolicy(env, **kwargs)
    elif algo_type == "HPCDiv":
        return HPCDivPolicy(env, **kwargs)
    elif algo_type == "ppo":
        from algos.torch.ppo import MlpPolicy
        n_steps, batch_size = 4096, 256
        if hasattr(env, "num_envs"):
            n_steps = int(n_steps / int(env.num_envs))
            batch_size = int(batch_size / int(env.num_envs))
        return PPO(MlpPolicy,
                   env=env,
                   n_steps=n_steps,
                   batch_size=batch_size,
                   gamma=0.99,
                   gae_lambda=0.95,
                   learning_rate=3e-4,
                   ent_coef=5e-4,
                   n_epochs=10,
                   ent_schedule=1.0,
                   clip_range=0.2,
                   verbose=verbose,
                   device=device,
                   tensorboard_log=log_dir,
                   policy_kwargs={'log_std_range': [None, 1.8],
                                  'net_arch': [{'pi': [32, 32], 'vf': [32, 32]}],
                                  },
                   **kwargs,
                   )

    elif algo_type == "sac":
        from algos.torch.sac import MlpPolicy
        return SAC(
            MlpPolicy,
            env=env,
            learning_rate=3e-4,
            batch_size=256,
            learning_starts=100,
            train_freq=(500, "step"),
            gradient_steps=1000,
            target_update_interval=1,
            gamma=0.99,
            ent_coef=0.01,
            device=device,
            verbose=verbose,
            tensorboard_log=log_dir,
            policy_kwargs={'net_arch': {'pi': [32, 32], 'qf': [32, 32]},
                           'optimizer_kwargs': {'betas': (0.9, 0.999)}},
            **kwargs
        )
    else:
        raise NameError("Not implemented policy name")
