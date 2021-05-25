import gym_envs
import numpy as np
import torch as th
from algos.torch.MaxEntIRL import RewardNet
from common.wrappers import *


def test_reward_wrapper():
    rwfn = RewardNet(inp=4, arch=[], feature_fn=lambda x: th.square(x)).double()
    env = RewardWrapper(gym_envs.make("IDP_custom-v2"), rwfn)
    env.reset()
    obs, rew, done, info = env.step(np.array([1, 1]))
