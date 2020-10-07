import os
from datetime import datetime
import numpy as np
import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines import PPO2

name = "ppo_ctl_1"
log_dir = "tmp/ctl/" + name
stats_dir = "tmp/ctl/" + name + ".pkl"
tensorboard_dir = os.path.join(os.path.dirname(__file__), "tmp", "log")
env_name = "IP_custom-v2"

env = make_vec_env(env_name)
env = VecNormalize(env, norm_reward=True, clip_reward=np.inf, norm_obs=False)

policy_kwargs = dict(net_arch=[dict(pi=[32], vf=[64, 32])])

model = PPO2("MlpPolicy",
             tensorboard_log=tensorboard_dir,
             verbose=1,
             env=env,
             gamma=0.96,
             n_steps=3000,
             lam=0.96,
             policy_kwargs=policy_kwargs)

model.learn(total_timesteps=600000, tb_log_name=name)

model.save(log_dir)
stats_path = os.path.join(stats_dir)
env.save(stats_path)

now = datetime.now()
print("%s.%s.%s. %s:%s" %(now.year, now.month, now.day, now.hour, now.minute))