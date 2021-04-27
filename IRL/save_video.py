import gym_envs
import os
import torch

from IRL.project_policies import def_policy
from algo.torch.ppo import PPO
from common.wrappers import CostWrapper
from common.verification import verify_policy, video_record
from common.util import make_env
from scipy import io

if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "ppo"
    name = "{}/{}/2021-3-19-18-27-54".format(env_type, algo_type)
    num = 20
    model_dir = os.path.join("tmp", "log", name, "model")
    costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
    algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
    # algo = PPO.load(model_dir + "/extra_ppo")
    env = make_env("{}_custom-v1".format(env_type), use_vec_env=False, sub="sub01", n_steps=600)
    # env = CostWrapper(gym_envs.make("{}_custom-v1".format(env_type), n_steps=200), costfn)
    exp = def_policy(env_type, env)
    dt = env.dt

    imgs = []
    for i in range(5):
        act_list, st_list, frames = verify_policy(env, algo, 'rgb_array')
        pltq_list = st_list[:, 4:]
        obs_list = st_list[:, :4]
        imgs += frames
    if imgs[0] is not None:
        video_record(imgs, "videos/{}_agent.avi".format(name), dt)
    env.close()
