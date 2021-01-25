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

class LocalCW(CostWrapper):
    def reward(self, obs):
        cost_inp = torch.from_numpy(obs).to(self.costfn.device)
        return -1e20*self.costfn.forward(cost_inp)


name = "2021-1-22-15-20-34"
num = 50
model_dir = os.path.join("..", "tmp", "log", name, "model")
costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
# algo = PPO.load(model_dir + "/extra_ppo.zip")
algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
env = LocalCW(gym.make("IP_custom-v1", n_steps=100), costfn)
exp = ExpertPolicy(env)
dt = env.dt
q = np.array([0.10, 0.06])
imgs1, imgs2 = [], []

for _ in range(1):
    obs = env.reset()
    done = False
    env.set_state(np.array([q[0]]), np.array([q[1]]))
    # env.render()
    while not done:
        act, _ = algo.predict(obs, deterministic=False)
        obs, rew, done, info = env.step(act)
        # env.render()
        time.sleep(dt)