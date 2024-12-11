import time
import os
import numpy as np
from scipy import signal

from imitation.data.rollout import make_sample_until

from algos.torch.ppo import PPO
from algos.torch.OptCont import DiscreteLQRPolicy
from common.sb3.util import make_env
from common.sb3.rollouts import generate_trajectories_without_shuffle


class IDPLQRPolicy(DiscreteLQRPolicy):
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
        # self.B[2:, :] = np.array([[1, -7.7], [0, 24.5]])
        self.A, self.B, _, _, dt = signal.cont2discrete((self.A, self.B, np.array([1, 1, 1, 1]), 0), self.env.envs[0].dt)
        # self.max_t = self.env.get_attr("spec")[0].max_episode_steps // 2
        self.max_t = 400
        self.Q = np.diag([0.7139, 0.5872182, 1.0639979, 0.9540204])
        self.R = np.diag([.061537065, .031358577])
        self.gear = 100


def test_rollout_fn():
    venv = make_env(f"IDP_custom-v1", num_envs=25)
    agent = IDPLQRPolicy(venv, alpha=0.02)
    sample_until = make_sample_until(n_timesteps=None, n_episodes=250)
    t1 = time.time()
    trajectories = generate_trajectories_without_shuffle(agent, venv, sample_until, deterministic_policy=False)
    print(time.time() - t1)


def test_custom_rollout():
    agent_path = os.path.join("..", "..", "RL", "HPC", "tmp", "log", "HPC_custom", "ppo", "policies_4", "agent")
    subpath = os.path.join("..", "..", "IRL", "demos", "HPC", "sub01", "sub01")
    venv = make_env("HPC_custom-v0", num_envs=5, subpath=subpath, wrapper="ActionWrapper")
    agent = PPO.load(agent_path)
    agent.set_env(venv)
    ob = venv.reset()
    obs = ob[..., np.newaxis]
    dones = np.array([False for _ in range(5)])
    while not dones.any():
        acts, _ = agent.predict(ob, deterministic=False)
        ob, _, dones, infos = venv.step(acts)
        obs = np.concatenate([obs, ob[..., np.newaxis]], axis=2)

    print('end')
