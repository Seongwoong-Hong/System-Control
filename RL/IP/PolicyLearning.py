import gym
import os

from stable_baselines3.common import callbacks
from imitation.policies import serialize

from algos.torch.ppo import PPO, MlpPolicy
from common.callbacks import VideoCallback
from common.util import make_env
from mujoco_py import GlfwContext


if __name__ == "__main__":
    env_type = "IP"
    algo_type = "ppo"
    name = f"{env_type}/{algo_type}"
    env_id = "IP_custom-v1"
    device = "cpu"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, "tmp", "log", name)
    os.makedirs(log_dir, exist_ok=True)
    GlfwContext(offscreen=True)
    env = make_env(env_id, num_envs=8)
    algo = PPO(MlpPolicy,
               env=env,
               n_steps=256,
               batch_size=64,
               gamma=0.99,
               gae_lambda=0.95,
               learning_rate=3e-4,
               ent_coef=0.0,
               n_epochs=2,
               ent_schedule=1.0,
               clip_range=0.2,
               verbose=1,
               device=device,
               tensorboard_log=log_dir,
               policy_kwargs={'log_std_range': [-5, 5],
                              'net_arch': [{'pi': [32, 32], 'vf': [32, 32]}]},
               )
    n = 1
    while os.path.isdir(log_dir + f"/extra_{n}"):
        n += 1
    os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
    video_recorder = VideoCallback(gym.make(env_id),
                                   n_eval_episodes=5,
                                   render_freq=int(1e5))
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(5e5), save_policy_callback)
    callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    algo.learn(total_timesteps=int(1e6), tb_log_name="extra", callback=callback_list)
    algo.save(log_dir+f"/policies_{n}/{algo_type}0")
    print(f"saved as policies_{n}/{algo_type}0")
