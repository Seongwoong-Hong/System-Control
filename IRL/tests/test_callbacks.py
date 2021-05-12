import gym_envs
import os

from algo.torch.ppo import PPO, MlpPolicy

from imitation.policies import serialize
from stable_baselines3.common import callbacks


def test_save_policies():
    env = gym_envs.make("IDP_custom-v2")
    policy_dir = os.path.join("..", "tmp", "log", "ppo", "IDP", "AIRL", "tests")
    algo = PPO(MlpPolicy, env=env, tensorboard_log=policy_dir, verbose=1)
    save_policy_callback = serialize.SavePolicyCallback(policy_dir, None)
    save_policy_callback = callbacks.EveryNTimesteps(
        2500, save_policy_callback
    )
    algo.learn(total_timesteps=int(5000), tb_log_name="extra", callback=save_policy_callback)