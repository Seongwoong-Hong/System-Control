import gym, gym_envs
import numpy as np
from imitation.data import rollout, types
from stable_baselines3.common.vec_env import DummyVecEnv
from algo.torch.OptCont.policies import LQRPolicy

# for the IP environment
class IPPolicy(LQRPolicy):
    def _build_env(self):
        m, h, I, g = 5.0, 0.5, 1.667, 9.81
        Q = np.array([[1, 0], [0, 1]])
        R = 0.001*np.array([[1]])
        A = np.array([[0, 1],
                      [m*g*h/I, 0]])
        B = np.array([[0], [1/I]])
        return A, B, Q, R

# for the IDP environment
class IDPPolicy(LQRPolicy):
    def _build_env(self):
        m1, m2, h1, h2, I1, I2, g = 5.0, 5.0, 0.5, 0.5, 1.667, 1.667, 9.81
        Q = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        R = 0.0001*np.array([[1, 0],
                             [0, 1]])
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [m1*g*h1/I1, 0, 0, 0],
                      [0, m2*g*h2/I2, 0, 0]])
        B = np.array([[0, 0], [0, 0],
                      [1/I1, -1/I1], [0, 1/I2]])
        return A, B, Q, R

def def_policy(env_type, env):
    if env_type == "IP":
        return IPPolicy(env)
    elif env_type == "IDP":
        return IDPPolicy(env)
    else:
        raise NameError("Not defined policy name")

if __name__ == "__main__":
    n_steps, n_episodes = 100, 30
    env_type = "IP"
    env_name = "{}_custom-v0".format(env_type)
    env = gym.make(env_name, n_steps=n_steps)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    ExpertPolicy = def_policy(env_type, env)
    trajectories = rollout.generate_trajectories(ExpertPolicy, DummyVecEnv([lambda: env]), sample_until, deterministic_policy=False)
    save_name = "demos/{}/expert.pkl".format(env_type)
    types.save(save_name, trajectories)
    print("Expert Trajectories are saved")