import gym, gym_envs, torch, os, pickle, time
import numpy as np
from algo.torch.OptCont import LQRPolicy
from imitation.data.rollout import flatten_trajectories
from matplotlib import pyplot as plt

class ExpertPolicy(LQRPolicy):
    def _build_env(self):
        m, g, h, I = 5.0, 9.81, 0.5, 1.667
        self.Q = np.array([[1, 0], [0, 1]])
        self.R = 0.001
        self.A = np.array([[0, m*g*h/I], [1, 0]])
        self.B = np.array([[1/I], [0]])
        return self.A, self.B, self.Q, self.R

model_dir = os.path.join("..", "tmp", "model")
env = gym.make("IP_custom-v2")
exp = ExpertPolicy(env)
expert_dir = os.path.join("..", "demos", "expert_bar_100.pkl")
with open(expert_dir, "rb") as f:
    expert_trajs = pickle.load(f)
    learner_trajs = []

obs_list, cost_list = [], []
Q, R, dt = exp.Q, exp.R, env.dt
env.reset()

for traj in expert_trajs:
    tran = flatten_trajectories([traj])
    env.set_state(np.array([tran[0]['obs'][1]]), np.array([tran[0]['obs'][0]]))
    obs = env._get_obs()
    obs_list.append(obs.reshape(-1))
    # env.render()
    for t in range(len(tran)):
        act = tran[t]['acts']
        obs, _, _, _ = env.step(act)
        cost = obs @ Q @ obs.T + act*act*R * exp.gear**2
        cost_list.append(cost.reshape(-1))
        obs_list.append(obs.reshape(-1))
        # env.render()
        time.sleep(dt)

print('end')