import gym, gym_envs
import numpy as np
from imitation.data import rollout, types
from stable_baselines3.common.vec_env import DummyVecEnv
from algo.torch.OptCont.policies import LQRPolicy

class ExpertPolicy(LQRPolicy):
    def _build_env(self):
        m, g, h, I = 5.0, 9.81, 0.5, 1.667
        Q = np.array([[1, 0], [0, 1]])
        R = 0.001
        A = np.array([[0, m*g*h/I], [1, 0]])
        B = np.array([[1/I], [0]])
        return A, B, Q, R

if __name__ == "__main__":
    n_steps, n_episodes = 100, 30
    env = gym.make("IP_custom-v2", n_steps=n_steps)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    ExpertPolicy = ExpertPolicy(env)
    trajectories = rollout.generate_trajectories(ExpertPolicy, DummyVecEnv([lambda: env]), sample_until, deterministic_policy=False)
    types.save("demos/IP/expert_broad.pkl", trajectories)
    print("Expert Trajectories are saved")