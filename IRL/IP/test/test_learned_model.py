import os, torch, gym, gym_envs
from algo.torch.ppo import PPO
from algo.torch.IRL import RewfromMat
from matplotlib import pyplot as plt

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation), done, info

    def reward(self, observation):
        return self.rwfn.forward(torch.from_numpy(observation).to(rwfn.device))

model_dir = os.path.join("..", "tmp", "model")
rwfn = torch.load(model_dir + "/f_rwfn.pt")
algo = PPO.load(model_dir + "/f_ppo.zip")
env = RewardWrapper(gym.make("IP_custom-v2", n_steps=40), rwfn)

rew_list = []

obs = env.reset()
env.render()
done = False
while not done:
    act, _ = algo.predict(obs, deterministic=True)
    obs, rew, done, info = env.step(act)
    env.render()
    rew_list.append(rew.item())

plt.plot(rew_list)
plt.show()