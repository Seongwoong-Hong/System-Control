import gym_envs
import numpy as np
import torch as th
from algos.torch.MaxEntIRL import RewardNet
from common.wrappers import *
from common.util import make_env


def test_reward_wrapper():
    rwfn = RewardNet(inp=4, arch=[], feature_fn=lambda x: x.square(), use_action_as_inp=True).double()
    env = make_env("HPC_custom-v1", use_vec_env=True, num_envs=5)
    wenv = RewardWrapper(env, rwfn)
    wenv.reset()
    obs, rew, done, info = wenv.step(wenv.action_space.sample())


def test_obs_concat():
    def feature_fn(x):
        return th.cat([x, x.square()], dim=1)
    env = make_env("HPC_custom-v1", subpath="../../IRL/demos/HPC/sub01/sub01")
    timesteps = 4
    env.reset()
    obs = env.observation_space.sample()
    inp = feature_fn(th.from_numpy(obs.reshape(1, -1))).shape[1]
    rwfn = RewardNet(inp=inp*timesteps, arch=[], feature_fn=feature_fn, use_action_as_inp=False, device='cpu').double()
    wrap_env = ActionRewardWrapper(env, rwfn, timesteps)
    a = wrap_env.action_space.sample()
    _, _, _, _ = wrap_env.step(a)
