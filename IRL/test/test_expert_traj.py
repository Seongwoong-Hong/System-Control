import gym, gym_envs, torch, os, pickle, time
import numpy as np
from imitation.data.rollout import flatten_trajectories
from matplotlib import pyplot as plt
from IRL.project_policies import def_policy

type = "IP"
env = gym.make("{}_custom-v0".format(type))
exp = def_policy(type, env)
expert_dir = os.path.join("..", "demos", type, "expert.pkl")
with open(expert_dir, "rb") as f:
    expert_trajs = pickle.load(f)
    learner_trajs = []

obs_list, cost_list = [], []
Q, R, dt = exp.Q, exp.R, env.dt
nq = env.model.nq
env.reset()

for traj in expert_trajs:
    tran = flatten_trajectories([traj])
    env.set_state(np.array([tran[0]['obs'][:nq]]).reshape(env.model.nq), np.array([tran[0]['obs'][nq:]]).reshape(env.model.nv))
    obs = env._get_obs()
    obs_list.append(obs.reshape(-1))
    env.render()
    for t in range(len(tran)):
        act = tran[t]['acts']
        obs, _, _, _ = env.step(act)
        cost = obs @ Q @ obs.T + act @ R @ act.T * exp.gear**2
        cost_list.append(cost.reshape(-1))
        obs_list.append(obs.reshape(-1))
        env.render()
        time.sleep(dt)

print('end')