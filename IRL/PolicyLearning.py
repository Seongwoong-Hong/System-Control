import gym_envs
import os
import torch

from stable_baselines3.common.vec_env import DummyVecEnv

from algo.torch.ppo import PPO, MlpPolicy
from common.callbacks import VFCustomCallback
from common.wrappers import CostWrapper


if __name__ == "__main__":
    env_type = "IDP"
    name = "{}/2021-2-5-20-33-56".format(env_type)
    num = 2

    model_dir = os.path.join("tmp", "log", name, "model")
    costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
    env_id = "{}_custom-v1".format(env_type)
    n_steps = 200
    device = "cpu"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, "tmp", "log", name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    env = DummyVecEnv([lambda: CostWrapper(gym_envs.make(env_id, n_steps=n_steps), costfn)])
    # env = DummyVecEnv([lambda: gym_envs.make(env_id, n_steps=n_steps)])
    algo = PPO(MlpPolicy,
               env=env,
               n_steps=4096,
               batch_size=128,
               gamma=0.99,
               gae_lambda=0.95,
               clip_range=0.2,
               ent_coef=0.015,
               verbose=1,
               device=device,
               tensorboard_log=log_dir,
               policy_kwargs={'log_std_range': [-5, 1]},
               )
    video_recorder = VFCustomCallback(log_dir + "/video/" + name,
                                      gym_envs.make(env_id, n_steps=n_steps),
                                      n_eval_episodes=5,
                                      render_freq=1024000,
                                      costfn=costfn)
    algo.learn(total_timesteps=1024000, callback=video_recorder, tb_log_name="extra")
    algo.save(model_dir+"/extra_ppo")
