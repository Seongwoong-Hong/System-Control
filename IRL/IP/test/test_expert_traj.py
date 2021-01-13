import gym, gym_envs, torch, os, pickle, time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from algo.torch.IRL import RewfromMat
from algo.torch.ppo import PPO
from common.rollouts import get_trajectories_probs
from matplotlib import pyplot as plt

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation), done, info

    def reward(self, observation):
        return self.rwfn.forward(torch.from_numpy(observation).to(self.rwfn.device))

model_dir = os.path.join("..", "tmp", "model")
rwfn = torch.load(model_dir + "/f_rwfn.pt")
algo = PPO.load(model_dir + "/f_ppo.zip")
env = DummyVecEnv([lambda: gym.make("IP_custom-v2", n_steps=400)])
expert_dir = os.path.join("..", "demos", "expert_bar_40.pkl")
with open(expert_dir, "rb") as f:
    expert_trajs = pickle.load(f)
    learner_trajs = []

obs_list = []
env.reset()
expert_trans = get_trajectories_probs(expert_trajs, algo.policy)
for tran in expert_trans:
    env.env_method('set_state', np.array([tran[0]['obs'][1]]), np.array([tran[0]['obs'][0]]))
    obs = env.env_method('_get_obs')
    obs_list.append(obs[0].reshape(-1))
    env.render()
    for t in range(len(tran)):
        obs, _, _, _ = env.step(tran[t]['acts'])
        obs_list.append(obs.reshape(-1))
        env.render()
        time.sleep(0.05)

print('end')