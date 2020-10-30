import gym, gym_envs, os
import numpy as np

from RL.algo.torch.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        action *= 0.125 * (self.action_space.high - self.action_space.low)
        return np.clip(action, self.action_space.low, self.action_space.high)

def make_env(env_id, rank, Wrapper_class = None, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, max_ep=1000)
        env.seed(seed + rank)
        if Wrapper_class is not None:
            env = Wrapper_class(env)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    name = "ppo_ctl_1"
    log_dir = "tmp/IP_ctl/torch/" + name
    stats_dir = "tmp/IP_ctl/torch/" + name + ".pkl"
    tensorboard_dir = os.path.join(os.path.dirname(__file__), "tmp", "log", "torch")
    env_id = "CartPoleCont-v0"
    num_cpu = 10  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i, NormalizedActions) for i in range(num_cpu)])
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    model = PPO('MlpPolicy',
             tensorboard_log=tensorboard_dir,
             verbose=1,
             env=env,
             gamma=1,
             n_steps=6400,
             gae_lambda=1,
             policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=3200000, tb_log_name=name)

    model.save(log_dir)
    stats_path = os.path.join(stats_dir)