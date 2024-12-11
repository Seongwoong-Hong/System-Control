import os
from common.sb3.util import make_env
from algos.torch.sac import SAC, MlpPolicy


def test_save():
    env = make_env("IDP_custom-v2")
    agent = SAC(MlpPolicy, env)
    current_path = os.path.dirname(__file__)
    agent.learn(total_timesteps=100)
    agent.save(current_path + "/agent")
