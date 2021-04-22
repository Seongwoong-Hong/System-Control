import gym_envs
import numpy as np
import os
import pickle
import time
from imitation.data.rollout import flatten_trajectories

from IRL.project_policies import def_policy

env_type = "HPC"
env = gym_envs.make("{}_custom-v0".format(env_type), n_steps=600)
exp = def_policy(env_type, env)
expert_dir = os.path.join("..", "demos", env_type, "lqrTest.pkl")
with open(expert_dir, "rb") as f:
    expert_trajs = pickle.load(f)
    learner_trajs = []

obs_list, act_list = [], []
Q, R, dt = exp.Q, exp.R, env.dt
nq, nv = env.model.nq, env.model.nv

for traj in expert_trajs:
    env.reset()
    tran = flatten_trajectories([traj])
    env.pltq = tran.obs[:, 4:]
    # pltqs = [tran.obs[:, -2:]]
    # env.set_pltqs(pltqs)
    # env.set_state(np.array([tran[0]['obs'][:nq]]).reshape(nq),
    #               np.array([tran[0]['obs'][nq:nq+nv]]).reshape(nv))
    # obs = env._get_obs()
    # obs_list.append(obs.reshape(-1))
    # env.render()
    for t in range(len(tran)):
        act = tran[t]['acts']
        obs_data = tran[t]['obs']
        # env.set_state(obs_data[:nq].reshape(nq),
        #               obs_data[nq:].reshape(nv))
        obs, _, _, _ = env.step(act)
        # cost = obs @ Q @ obs.T + act @ R @ act.T * exp.gear**2
        act_list.append(act.reshape(-1))
        obs_list.append(obs.reshape(-1))
        env.render()
        time.sleep(dt)


env.close()
print('end')
