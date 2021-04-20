import gym_envs
import os
import torch

from IRL.project_policies import def_policy
from algo.torch.ppo import PPO
from common.wrappers import CostWrapper
from common.verification import verify_policy, video_record
from scipy import io


env_type = "HPC"
algo_type = "ppo"
name = "{}/{}/2021-3-19-18-27-54".format(env_type, algo_type)
num = 20
model_dir = os.path.join("tmp", "log", name, "model")
costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
# algo = PPO.load(model_dir + "/extra_ppo")
sub = "sub01"
pltqs = []
for i in range(35):
    file = os.path.join("demos", env_type, sub, sub + "i%d.mat" % (i+1))
    pltqs += [io.loadmat(file)['pltq']]

env = gym_envs.make("{}_custom-v1".format(env_type), n_steps=600, pltqs=pltqs)
# env = CostWrapper(gym_envs.make("{}_custom-v1".format(env_type), n_steps=200), costfn)
exp = def_policy(env_type, env)
dt = env.dt

for i in range(5):
    # act1_list, st1_list = according_policy(env, exp)
    act_list, st_list, imgs = verify_policy(env, algo, 'rgb_array')
    pltq_list = st_list[:, 4:]
    obs_list = st_list[:, :4]

video_record(imgs, "videos/{}_agent.avi".format(name), dt)
env.close()
