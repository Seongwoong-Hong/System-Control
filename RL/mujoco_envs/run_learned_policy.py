import gym
import os

from common.verification import verify_policy
from algo.torch.ppo import PPO

if __name__ == "__main__":
    env_type = "Hopper"
    algo_type = "ppo"
    name = f"{env_type}/{algo_type}/Hopper_v3"
    env_id = "Hopper-v3"
    device = "cpu"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, "tmp", "log", name)
    env = gym.make(env_id)
    algo = PPO.load(log_dir + "/ppo0.zip")
    verify_policy(env, algo)