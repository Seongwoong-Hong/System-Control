import gym, gym_envs
import torch as th
import numpy as np
from scipy import linalg
from imitation.data import rollout, types
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv

class ExpertPolicy(BasePolicy):
    def __init__(self,
                 observation_space:gym.spaces.Space,
                 action_space:gym.spaces.Space):
        super(ExpertPolicy, self).__init__(
            observation_space,
            action_space)
        m, g, h, I = 43.16, 9.81, 0.966, 1.4795
        A = np.array([[0, m*g*h/I], [1, 0]])
        B = np.array([[1/I], [0]])
        Q = np.array([[1, 0], [0, 1]])
        R = 0.001
        X = linalg.solve_continuous_are(A, B, Q, R)
        K = (1/R * (B.T @ X)).reshape(-1)
        self.D, self.P = K[0], K[1]

    def forward(self):
        return None

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return -(self.P*observation[0][1] + self.D*observation[0][0]).reshape(1, 1)

if __name__ == "__main__":
    n_steps, n_episodes = 40, 10
    env = DummyVecEnv([lambda: gym.make("IP_custom-v2", n_steps=n_steps)])
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    ExpertPolicy = ExpertPolicy(env.observation_space, env.action_space)
    trajectories = rollout.generate_trajectories(ExpertPolicy, env, sample_until, deterministic_policy=True)
    types.save("demos/expert_40.pkl", trajectories)
    print("Expert Trajectories are saved")