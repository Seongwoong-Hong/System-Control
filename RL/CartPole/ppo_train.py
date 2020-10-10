import os
import tensorflow as tf
from datetime import datetime
import numpy as np
import gym
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common import make_vec_env, SetVerbosity, tf_util
from stable_baselines import PPO2

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        action *= 0.5 * (self.action_space.high - self.action_space.low)
        return np.clip(action, self.action_space.low, self.action_space.high)

name = "ppo_ctl_2"
log_dir = "tmp/IP_ctl/" + name
stats_dir = "tmp/IP_ctl/" + name + ".pkl"
tensorboard_dir = os.path.join(os.path.dirname(__file__), "tmp", "log")
env_name = "CartPoleCont-v0"

env = make_vec_env(env_name, n_envs=5, wrapper_class=NormalizedActions)
# Automatically normalize the input features and reward
env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=10.)

policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

model = PPO2("MlpPolicy",
             tensorboard_log=tensorboard_dir,
             verbose=1,
             env=env,
             gamma=0.99,
             n_steps=1000,
             lam=0.9,
             policy_kwargs=policy_kwargs)

model.learn(total_timesteps=10000000, tb_log_name=name)

model.save(log_dir)
stats_path = os.path.join(stats_dir)
env.save(stats_path)

now = datetime.now()
print("%s.%s.%s., %s:%s" %(now.year, now.month, now.day, now.hour, now.minute))