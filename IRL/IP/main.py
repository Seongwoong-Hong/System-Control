import torch, gym, gym_envs, pickle, random
from stable_baselines3.common.vec_env import DummyVecEnv
from algo.torch.ppo import PPO
from imitation.data import rollout
from torch import nn

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation), done, info

    def reward(self, observation):
        return self.rwfn.forward(torch.from_numpy(observation).float())

class RewfromMat(nn.Module):
    def __init__(self, inp):
        super(RewfromMat, self).__init__()
        self.layer1 = nn.Linear(inp, 1)

    def forward(self, obs):
        return self.layer1(obs)

def cal_sample_cost(sampleL, sampleE, rwfn):
    L_trans = rollout.flatten_trajectories(sampleL)
    E_trans = rollout.flatten_trajectories(sampleE)
    IOCLoss = 0
    for i in range(E_trans.__len__()):
        IOCLoss += rwfn(torch.from_numpy(E_trans[i]['obs']).float())
    IOCLoss /= E_trans.__len__()
    # for i in range(L_trans.__len__()):
        # IOCLoss +=
    return IOCLoss

if __name__ == "__main__":
    n_steps, n_episodes = 200, 10
    env = gym.make("IP_custom-v2", n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    rwfn = RewfromMat(num_obs).float()
    env = DummyVecEnv([lambda: env])
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    algo = PPO("MlpPolicy",
               env=env,
               n_steps=n_steps,
               batch_size=200,
               gamma=0.99,
               gae_lambda=0.95,
               ent_coef=0.05,
               verbose=1,
               device='cpu')
    with open("demos/expert.pkl", "rb") as f:
        trajectories = pickle.load(f)
    optimizer = torch.optim.Adam(rwfn.parameters(), lr=1e-5)
    for _ in range(10):
        # update cost function
        for k in range(10):
            with torch.no_grad():
                sampleT = rollout.generate_trajectories(algo.policy, env, sample_until) #sample trajectory
            sampleD = random.sample(trajectories, 5) #Expert Demo trajectory
            # todo: cal_sample_cost: calculate IOC Loss using expert and experienced trajectories and rew func.
            IOCLoss = cal_sample_cost(sampleT, sampleD, rwfn) #calculate sample's cost again
            optimizer.zero_grad()
            IOCLoss.backward()
            optimizer.step()
        with torch.no_grad():
            env = DummyVecEnv([lambda: RewardWrapper(env, rwfn)])
            algo.set_env(env)
            algo.learn(total_timesteps=100)
