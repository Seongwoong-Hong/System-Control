import torch, gym, gym_envs, pickle, random
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
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
        return self.rwfn.forward(torch.from_numpy(observation).float())

class RewfromMat(nn.Module):
    def __init__(self, inp):
        super(RewfromMat, self).__init__()
        self.layer1 = nn.Linear(inp, 1)

    def forward(self, obs):
        return self.layer1(obs)

def generate_trajectories(
    policy,
    venv: VecEnv,
    sample_until: rollout.GenTrajTerminationFn,
    *,
    deterministic_policy: bool = False,
    rng: np.random.RandomState = np.random,
) -> Sequence[types.TrajectoryWithRew]:

    get_action = policy.forward
    if isinstance(policy, BaseAlgorithm):
        policy.set_env(venv)

    trajectories = []
    trajectories_accum = rollout.TrajectoryAccumulator()
    obs = venv.reset()
    for env_idx, ob in enumerate(obs):
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    active = np.ones(venv.num_envs, dtype=np.bool)
    while np.any(active):
        th_obs = torch.as_tensor(obs)
        acts, _, log_probs = get_action(th_obs)
        obs, rews, dones, infos = venv.step(acts)
        infos[0]['log_probs'] = log_probs
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts, obs, rews, dones, infos
        )
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            active &= ~dones
    rng.shuffle(trajectories)

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories

def cal_sample_cost(sampleL, sampleE, rwfn):
    E_trans = rollout.flatten_trajectories(sampleE)
    IOCLoss, w = 0, 0
    for i in range(E_trans.__len__()):
        IOCLoss -= rwfn(torch.from_numpy(E_trans[i]['obs']).float())
    IOCLoss /= sampleE.__len__()
    for L_traj in sampleL:
        c, log_q = 0, 0
        trans = rollout.flatten_trajectories([L_traj])
        for i in range(trans.__len__()):
            c -= rwfn(torch.from_numpy(trans[i]['obs']).float())
            log_q += trans[i]['infos']['log_probs']
        w += torch.exp(-c)/torch.exp(log_q)
    IOCLoss += torch.log(w/sampleE.__len__())
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
                sampleT = generate_trajectories(algo.policy, env, sample_until) #sample trajectory
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
