import gym
import gym_envs
import os
import time
import torch

from IRL.project_policies import def_policy
from algo.torch.ppo import PPO
from common.wrappers import CostWrapper


class LocalCW(CostWrapper):
    def reward(self, observation):
        cost_inp = torch.from_numpy(observation).to(self.costfn.device)
        return -1e20*self.costfn.forward(cost_inp)


env_type = "IDP"
name = "{}/2021-1-25-15-19-30".format(env_type)
num = 14
model_dir = os.path.join("..", "tmp", "log", name, "model")
costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
# algo = PPO.load(model_dir + "/test_ppo.zip")
# algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
env = CostWrapper(gym.make("{}_custom-v1".format(env_type), n_steps=200), costfn)
exp = def_policy(env_type, env)
dt = env.dt

for _ in range(5):
    done = False
    obs = env.reset()
    env.render()
    while not done:
        act, _ = exp.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        env.render()
        time.sleep(dt)

