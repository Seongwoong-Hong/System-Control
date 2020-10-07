import os
import tensorflow as tf
from datetime import datetime

import gym
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common import make_vec_env, SetVerbosity, tf_util
from stable_baselines import PPO2

name = "ppo_ctl_1"
log_dir = "tmp/IP_ctl/" + name
stats_dir = "tmp/IP_ctl/" + name + ".pkl"
tensorboard_dir = os.path.join(os.path.dirname(__file__), "tmp", "log")
env_name = "CartPoleCont-v0"

env = make_vec_env(env_name)
# Automatically normalize the input features and reward
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)

policy_kwargs = dict(net_arch=[dict(pi=[64, 32], vf=[64, 32])])

model = PPO2("MlpPolicy",
             tensorboard_log=tensorboard_dir,
             verbose=1,
             env=env,
             gamma=0.96,
             n_steps=1000,
             lam=0.96,
             policy_kwargs=policy_kwargs)

model.learn(total_timesteps=600000, tb_log_name=name)

model.save(log_dir)
stats_path = os.path.join(stats_dir)
env.save(stats_path)

now = datetime.now()
print("%s.%s.%s. %s:%s" %(now.year, now.month, now.day, now.hour, now.minute))