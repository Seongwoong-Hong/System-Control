import torch, gym, gym_envs, pickle, random, copy
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from algo.torch.ppo import PPO
from imitation.data import rollout, types
from torch import nn
from typing import Sequence

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation), done, info

    def reward(self, observation):
        return self.rwfn.forward(torch.from_numpy(observation).to(rwfn.device))

class RewfromMat(nn.Module):
    def __init__(self,
                 inp,
                 device = 'cpu',
                 optimizer_class=torch.optim.Adam,
                 lr = 1e-4):
        super(RewfromMat, self).__init__()
        self.device = device
        self.layer1 = nn.Linear(inp, 2*inp)
        self.layer2 = nn.Linear(2*inp, 1)
        self.relu = nn.ReLU()
        self.optimizer_class = optimizer_class
        self._build(lr)

    def _build(self, lr):
        self.optimizer = self.optimizer_class(self.parameters(), lr)

    def sample_trajectory_sets(self, learner_trans, expert_trans):
        self.sampleL = random.sample(learner_trans, 10)
        self.sampleE = random.sample(expert_trans, 5)

    def forward(self, obs):
        out = self.layer1(obs)
        return self.layer2(out)

    def learn(self, epoch):
        self.train()
        for _ in range(epoch):
            IOCLoss = 0.0
            for E_trans in self.sampleE:
                for i in range(len(E_trans)):
                    IOCLoss -= self.forward(torch.from_numpy(E_trans[i]['obs']).to(self.device))
            IOCLoss /= len(self.sampleE)
            for trans_i in self.sampleE+self.sampleL:
                temp, cost = 0.0, 0.0
                for trans_j in self.sampleE+self.sampleL:
                    for t in range(len(trans_i)):
                        cost -= self.forward(torch.from_numpy(trans_i[t]['obs']).to(self.device))
                        with torch.no_grad():
                            temp += torch.exp(self.forward(torch.from_numpy(trans_j[t]['obs']).to(self.device)) \
                                            - self.forward(torch.from_numpy(trans_i[t]['obs']).to(self.device)) \
                                            + trans_i[t]['infos']['log_probs'] - trans_j[t]['infos']['log_probs'])
                IOCLoss -= cost / temp
            self.optimizer.zero_grad()
            IOCLoss.backward()
            self.optimizer.step()
        print("Loss: {:.2f}".format(IOCLoss.item()))
        return self

def get_trajectories_probs(
        trajectories,
        policy,
        rng: np.random.RandomState = np.random
) -> Sequence[types.TrajectoryWithRew]:
    transitions = []
    for traj in trajectories:
        trans = copy.deepcopy(rollout.flatten_trajectories_with_rew([traj]))
        for i in range(trans.__len__()):
            obs = trans[i]['obs'].reshape(1, -1)
            acts = trans[i]['acts'].reshape(1, -1)
            latent_pi, _, latent_sde = policy._get_latent(torch.from_numpy(obs).to(policy.device))
            distribution = policy._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
            log_probs = distribution.log_prob(torch.from_numpy(acts).to(policy.device))
            trans[i]['infos']['log_probs'] = log_probs
        transitions.append(trans)
    return transitions

if __name__ == "__main__":
    n_steps, n_episodes = 200, 10
    device = 'cuda'
    env_id = "IP_custom-v2"
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    rwfn = RewfromMat(num_obs, device=device).double().to(device)
    env = DummyVecEnv([lambda: env])
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    print("Start Guided Cost Learning...")
    print("Used environment is {}".format(env_id))
    algo = PPO("MlpPolicy",
               env=env,
               n_steps=2048,
               batch_size=128,
               gamma=0.99,
               gae_lambda=0.95,
               ent_coef=0.01,
               verbose=1,
               device=device)
    with open("demos/expert.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
        learner_trajs = []
    for _ in range(10):
        # update cost function
        for k in range(40):
            with torch.no_grad():
                learner_trajs += rollout.generate_trajectories(algo.policy, env, sample_until)
                expert_trans = get_trajectories_probs(expert_trajs, algo.policy)
                learner_trans = get_trajectories_probs(learner_trajs, algo.policy)
            rwfn.sample_trajectory_sets(learner_trans, expert_trans)
            rwfn.learn(epoch=50)

        # update policy
        env = DummyVecEnv([lambda: RewardWrapper(gym.make(env_id, n_steps=n_steps), rwfn.eval())])
        algo.set_env(env)
        algo.learn(total_timesteps=1024000)
