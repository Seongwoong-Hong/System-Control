import gym
import gym_envs
import os
import shutil

from algo.torch.ppo import PPO

if __name__ == "__main__":
    name = "test"
    log_dir = os.path.join(os.path.abspath("../tmp/log"), name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    model_dir = os.path.join(os.path.dirname(__file__), "tmp", "model")
    expert_dir = os.path.join(os.path.dirname(__file__), "demos", "expert_bar_100.pkl")
    shutil.copy(os.path.abspath("../../common/modules.py"), log_dir)
    shutil.copy(os.path.abspath("../../gym_envs/envs/IP_custom_exp.py"), log_dir)
    shutil.copy(os.path.abspath(__file__), log_dir)
    env = gym.make("IP_custom-v1")

    algo = PPO("MlpPolicy",
               env=env,
               verbose=1,
               tensorboard_log=log_dir)
    for _ in range(2):
        algo.learn(total_timesteps=1024, tb_log_name=name)
        algo.save("test")
