import gym_envs
import numpy as np
import os
import pickle
import time
from imitation.data.rollout import flatten_trajectories

from IRL.project_policies import def_policy

env_type = "HPC"
env = gym_envs.make("{}_custom-v0".format(env_type))
exp = def_policy(env_type, env)
expert_dir = os.path.join("..", "demos", env_type, "sub01.pkl")
with open(expert_dir, "rb") as f:
    expert_trajs = pickle.load(f)
    learner_trajs = []

obs_list, cost_list = [], []
Q, R, dt = exp.Q, exp.R, env.dt
nq = env.model.nq
env.reset()

for traj in expert_trajs:
    tran = flatten_trajectories([traj])
    env.set_state(np.array([tran[0]['obs'][:nq]]).reshape(env.model.nq),
                  np.array([tran[0]['obs'][nq:]]).reshape(env.model.nv))
    obs = env._get_obs()
    obs_list.append(obs.reshape(-1))
    # env.render()
    for t in range(len(tran)):
        act = tran[t]['acts']
        env.set_state(np.array([tran[t]['obs'][:nq]]).reshape(env.model.nq),
                      np.array([tran[t]['obs'][nq:]]).reshape(env.model.nv))
        obs, _, _, _ = env.step(act)
        cost = obs @ Q @ obs.T + act @ R @ act.T * exp.gear**2
        cost_list.append(cost.reshape(-1))
        obs_list.append(obs.reshape(-1))
        env.render()
        time.sleep(dt)

env.close()
print('end')
