import gym
import os

from stable_baselines3.common.vec_env import DummyVecEnv

from algo.torch.ppo import PPO, MlpPolicy
from common.callbacks import VideoCallback
from mujoco_py import GlfwContext


if __name__ == "__main__":
    env_type = "Hopper"
    algo_type = "ppo"
    name = f"{env_type}/{algo_type}/Hopper_v3"
    env_id = "Hopper-v3"
    device = "cuda:0"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, "tmp", "log", name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    GlfwContext(offscreen=True)
    env = DummyVecEnv([lambda: gym.make(env_id) for i in range(10)])
    algo = PPO(MlpPolicy,
               env=env,
               n_steps=10240,
               batch_size=1024,
               gamma=0.99,
               gae_lambda=0.95,
               learning_rate=3e-4,
               ent_coef=0.0,
               n_epochs=10,
               ent_schedule=1.0,
               clip_range=0.2,
               verbose=1,
               device=device,
               tensorboard_log=log_dir,
               policy_kwargs={'log_std_range': [-5, 5],
                              'net_arch': [{'pi': [128, 128], 'vf': [128, 128]}]},
               )
    video_recorder = VideoCallback(gym.make(env_id),
                                   n_eval_episodes=5,
                                   render_freq=71680)
    algo.learn(total_timesteps=7168000, tb_log_name="extra", callback=video_recorder)
    n = 0
    while os.path.isfile(log_dir+f"/{algo_type}{n}"):
        n += 1
    algo.save(log_dir+f"/{algo_type}{n}")
    print(f"saved as {algo_type}{n}")
