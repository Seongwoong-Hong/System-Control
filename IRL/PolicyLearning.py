import gym_envs
import os
import torch

from stable_baselines3.common.vec_env import DummyVecEnv

from IRL.project_policies import def_policy
from common.callbacks import VFCustomCallback
from mujoco_py import GlfwContext
from common.wrappers import CostWrapper


if __name__ == "__main__":
    env_type = "IDP"
    name = "{}/2021-2-7-14-45-45".format(env_type)
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
    GlfwContext(offscreen=True)
    # env = DummyVecEnv([lambda: CostWrapper(gym_envs.make(env_id, n_steps=n_steps), costfn)])
    env = DummyVecEnv([lambda: gym_envs.make(env_id, n_steps=n_steps)])
    algo = def_policy("ppo", env, device=device, log_dir=log_dir)
    video_recorder = VFCustomCallback(gym_envs.make(env_id, n_steps=n_steps),
                                      n_eval_episodes=5,
                                      render_freq=1024000,
                                      costfn=costfn)
    algo.learn(total_timesteps=1024000, callback=video_recorder, tb_log_name="extra")
    algo.save(model_dir+"/extra_ppo")
