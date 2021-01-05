import torch, gym, gym_envs, pickle
from algo.torch.ppo import PPO
from imitation.data import rollout
from torch import nn

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn

    def reward(self, observation):
        return self.rwfn.forward(observation)

class RewfromMat(nn.Module):
    def __init__(self, inp):
        super(RewfromMat, self).__init__()
        self.layer1 = nn.Linear(inp,1)

    def forward(self, obs):
        return self.layer1(obs)

if __name__ == "__main__":
    env = gym.make("IP_custom-v0")
    num_obs = env.observation_space.shape[0]
    rwfn = RewfromMat(num_obs)
    env = RewardWrapper(env, rwfn)
    n_steps = 4000
    n_episodes = 10
    sample_until = rollout.make_sample_until(n_steps, n_episodes)
    algo = PPO("MlpPolicy",
               env=env,
               n_steps=n_steps,
               batch_size=200,
               gamma=0.99,
               gae_lambda=0.95,
               ent_coef=0.05,
               verbose=1,
               device='cpu')
    with open("IRL/IP/demos/expert.pkl", "rb") as f:
        trajectories = pickle.load(f)
    transitions = rollout.flatten_trajectories(trajectories)
    for _ in range(10):
        # update cost function
        for k in range(10):
            sampleT = rollout.generate_trajectories(algo.policy, env, sample_until) #sample trajectory
            sampleD = transitions.sample() #Expert trajectory
            IOCLoss = cal_sample_cost(sampleT, sampleD) #calculate sample's cost again
            optimizer.zero_grad()
            IOCLoss.backward()
            optimizer.step()
        env = RewardWrapper(env, rwfn)
        algo.set_env(env)
        algo.learn(total_timesteps=100)
