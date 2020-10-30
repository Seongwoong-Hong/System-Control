#!/home/biomechserver/anaconda3/gym_custom/baseline/bin/python3

import gym, os, math
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from mujoco_py import cymj

name = "IP_ctl/ppo_ctl_1"
log_dir = "tmp/" + name + ".zip"
stats_dir = "tmp/" + name + ".pkl"
env_name = "IP_custom-v2"
stats_path = os.path.join(stats_dir)

max_steps = 3000

obs_result = []
rew_result = []
act_result = []
com_result = []

env = DummyVecEnv([lambda: gym.make(env_name)])
env = VecNormalize.load(stats_path, env)
env.env_method('set_state', np.array([0.15]), np.array([0.1]))
fname = "trajectory7.csv"
obs = env.env_method('_get_obs')[0]


for steps in range(max_steps):
	env.render()
	act = np.clip([-6.4*obs[0] - 1.8*obs[1]], -10, 10)
	obs = env.env_method('_get_obs')[0].tolist() + act.tolist()
	obs_result.append(obs)
	_, _, _, _ = env.step(act)
	action = env.get_attr('data')[0].qfrc_actuator
	act_result.append(action.squeeze().tolist())
	# com = [0.966 * math.sin(obs[0])] + [0.966 * math.cos(obs[0])]
	# com_result.append(com)

act = np.clip([-6.4*obs[0] - 1.8*obs[1]], -5, 5)
obs = env.env_method('_get_obs')[0].tolist() + act.tolist()
obs_result.append(obs)

print(act_result.__len__())
env.close()

from matplotlib import pyplot as plt
import csv

# obs_result = np.array(obs_result).squeeze()
# act_result = np.array(act_result).squeeze()
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.set_ylim(-0.3, 0.3)
# ax.plot(obs_result[:, 0])
# ax.plot(obs_result[:, 1])
# plt.title('Ankle state')
# plt.legend(['angle', 'velocity'])
# plt.show()
#
# plt.plot(act_result)
# plt.title('Ankle Actuation')
# plt.show()
#
# csvfile = open(fname, "w", newline="")
# csvwriter = csv.writer(csvfile)
# for row in obs_result:
# 	csvwriter.writerow(row)
#
# csvfile.close()