#!/home/biomechserver/anaconda3/gym_custom/baseline/bin/python3

import os
import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from matplotlib import pyplot as plt

name = "ctl/ppo_ctl_1"
log_dir = "tmp/" + name + ".zip"
stats_dir = "tmp/" + name + ".pkl"
env_name = "IP_custom-v2"
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
max_step = 3000

for _ in range(1):
    step = 0
    obs = env.reset()
    # obs = np.concatenate((env.get_attr('data')[0].qpos, env.get_attr('data')[0].qvel, env.get_attr('data')[0].qfrc_constraint))
    done = False
    while (not done) and bool(step < max_step):
        env.render("human")
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _ = env.step(act)
        action = env.get_attr('data')[0].qfrc_actuator
        act_result.append(action[0])
        obs_result.append(obs)
        step += 1

print("xxx")
print(act_result)