import gym_envs
import os
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from IRL.project_policies import def_policy
from common.callbacks import VFCustomCallback
from mujoco_py import GlfwContext
from common.wrappers import CostWrapper
from scipy import io


if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "ppo"
    name = "{}/{}/2021-2-9-16-19-31".format(env_type, algo_type)
    num = 2

    model_dir = os.path.join("tmp", "log", name, "model")
    # costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
    env_id = "{}_custom-v0".format(env_type)
    n_steps = 600
    device = "cpu"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, "tmp", "log", name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    GlfwContext(offscreen=True)
    sub = "sub01"
    expert_dir = os.path.join(current_path, "demos", env_type, sub + ".pkl")
    pltqs = []
    for i in range(35):
        file = os.path.join(current_path, "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
        pltqs += [io.loadmat(file)['pltq']]
    # env = DummyVecEnv([lambda: gym_envs.make(env_id, n_steps=n_steps, pltqs=pltqs)])
    env = SubprocVecEnv([lambda: gym_envs.make(env_id, n_steps=n_steps, pltqs=pltqs) for i in range(10)])
    algo = def_policy(algo_type, env, device=device, log_dir=log_dir)
    # video_recorder = VFCustomCallback(gym_envs.make(env_id, n_steps=n_steps),
    #                                   n_eval_episodes=5,
    #                                   render_freq=1024000,
    #                                   costfn=costfn)
    algo.learn(total_timesteps=7168000, tb_log_name="extra")
    algo.save(model_dir+"/extra_ppo")
