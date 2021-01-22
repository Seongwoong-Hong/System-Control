import os, torch, gym, gym_envs, time
import numpy as np
from algo.torch.ppo import PPO
from algo.torch.OptCont.policies import LQRPolicy
from common.wrappers import CostWrapper

class ExpertPolicy(LQRPolicy):
    def _build_env(self):
        m, g, h, I = 5.0, 9.81, 0.5, 1.667
        self.Q = np.array([[1, 0], [0, 1]])
        self.R = 0.001
        self.A = np.array([[0, m*g*h/I], [1, 0]])
        self.B = np.array([[1/I], [0]])
        return self.A, self.B, self.Q, self.R

name = "2021-1-22-11-3-31"
num = 10
model_dir = os.path.join("..", "tmp", "log", name, "model")
costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
algo = PPO.load(model_dir + "/extra_ppo.zip")
# algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
env = CostWrapper(gym.make("IP_custom-v1", n_steps=100), costfn)
exp = ExpertPolicy(env)
dt = env.dt
q = np.array([0.10, 0.06])
imgs1, imgs2 = [], []

for _ in range(1):
    obs = env.reset()
    done = False
    env.set_state(np.array([q[0]]), np.array([q[1]]))
    env.render()
    while not done:
        act, _ = algo.predict(obs, deterministic=False)
        obs, rew, done, info = env.step(act)
        env.render()
        # time.sleep(dt)