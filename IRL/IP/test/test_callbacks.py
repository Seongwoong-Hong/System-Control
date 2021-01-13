import gym, gym_envs, os

from stable_baselines3.common.vec_env import DummyVecEnv
from common.callbacks import VideoRecorderCallback
from algo.torch.ppo import PPO
from mujoco_py import GlfwContext

if __name__ == "__main__":
    n_steps, n_episodes = 40, 10
    env_id = "IP_custom-v2"
    log_dir = os.path.join("..", "tmp", "log")
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    env = DummyVecEnv([lambda: env])
    GlfwContext(offscreen=True)
    algo = PPO("MlpPolicy",
               env=env,
               n_steps=2048,
               batch_size=128,
               gamma=0.99,
               gae_lambda=0.95,
               ent_coef=0.01,
               verbose=1,
               device='cpu',
               tensorboard_log=log_dir)

    video_recorder = VideoRecorderCallback(log_dir+"/video/bar",
                                           gym.make(env_id, n_steps=n_steps),
                                           n_eval_episodes=5,
                                           render_freq=128)

    env = DummyVecEnv([lambda: gym.make(env_id, n_steps=n_steps)])
    algo.set_env(env)
    algo.learn(total_timesteps=2048, callback=video_recorder, tb_log_name="test_callback")