#!/home/biomechserver/anaconda3/envs/baseline/bin/python3

import os
import gym
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2, DDPG
from matplotlib import pyplot as plt

name = "IP_ctl/ppo_ctl_2"
log_dir = "tmp/" + name + ".zip"
stats_dir = "tmp/" + name + ".pkl"
env_name = "CartPoleCont-v0"
stats_path = os.path.join(stats_dir)

# Load the agent
model = PPO2.load(log_dir)

# Load the saved statistics
env = DummyVecEnv([lambda: gym.make(env_name)])
env = VecNormalize.load(stats_path, env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

obs_result = []
act_result = []
rew_result = []
coeff_result = []
max_step = 999

for _ in range(1):
    step = 0
    obs = env.reset()
    # env.env_method('set_state', np.array([0.1]), np.array([0.05]))
    # obs = env.normalize_obs(env.env_method('_get_obs'))
    # obs = np.concatenate((env.get_attr('data')[0].qpos, env.get_attr('data')[0].qvel, env.get_attr('data')[0].qfrc_constraint))
    done = False
    while (not done) and bool(step < max_step):
        # env.render("human")
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        action = info[0]['action']
        # coeff_result.append(act.squeeze().tolist())
        act_result.append(action.squeeze().tolist())
        rew_result.append(rew.squeeze().tolist())
        obs_result.append(env.get_original_obs().squeeze().tolist())
        step += 1
# for _ in range(10):
print("end")
import copy


# for i in range(100):
#     for j in range(100):
#         obs = env.reset()
#         env.env_method('set_state', np.array([0.2/100 * i]), np.array([-0.3 + 0.6/100 * i]))
#         ipos = copy.deepcopy(env.get_attr('data')[0].qpos)
#         ivel = copy.deepcopy(env.get_attr('data')[0].qvel)
#         nobs = env.normalize_obs(env.env_method('_get_obs'))
#         act, _ = model.predict(nobs, deterministic=True)
#         # nobs, rew, done, info = env.step(act)
#         th.append(ipos)
#         dth.append(ivel)
#         p.append(act[0][0])
#         d.append(act[0][1])
#         T.append(act[0][0]*ipos + act[0][1]*ivel)

        # print(ipos, ivel)
        # print(rew)
        # print(act)
# print(info[0]['act_reward'], info[0]['pos_reward'], info[0]['idx_reward'])

env.close()

# th = np.linspace(0,0.2,100)
# dth = np.linspace(-0.3,0.3,100)
# th,dth = np.meshgrid(dth,th)
#
# p = np.array(p).reshape(100,100)
# d = np.array(d).reshape(100,100)

from matplotlib import pyplot as plt

plt.plot(info['actu'])

# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# ax.plot_surface(dth, th, p, cmap='coolwarm', antialiased=False)
# plt.xlabel('th')
# plt.ylabel('dth')
# plt.title('p gain')
# plt.show()
# #
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(dth, th, d, cmap='coolwarm', antialiased=False)
# plt.xlabel('th')
# plt.ylabel('dth')
# plt.title('d gain')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(dth,p)
# ax.plot(dth,d)
# plt.legend(['p','d'])
# plt.show()

print('k')
# plt.plot(info[0]['action'])
# plt.title('Ankle Actuation')
# plt.show()

# obs_result = np.array(obs_result)
# act_result = np.array(act_result)
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
#
# plt.plot(rew_result)
# plt.title('Actuation Coefficient')
# plt.show()
#data.qfrc_actuator: actuator torque
