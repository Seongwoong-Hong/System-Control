import torch, gym, gym_envs, pickle, random, copy
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
        return self.rwfn.forward(torch.from_numpy(observation).double())

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
    E_trans = copy.deepcopy(rollout.flatten_trajectories(sampleE))
    IOCLoss, w = 0, 0
    for i in range(E_trans.__len__()):
        IOCLoss -= rwfn(torch.from_numpy(E_trans[i]['obs']).double())
    IOCLoss /= sampleE.__len__()
    for L_traj in sampleL:
        wi = 1
        trans = copy.deepcopy(rollout.flatten_trajectories([L_traj]))
        for i in range(trans.__len__()):
            wi *= torch.exp(-rwfn(torch.from_numpy(trans[i]['obs']).double()))/torch.exp(trans[i]['infos']['log_probs'])
        w += wi
    IOCLoss += torch.log(w/sampleE.__len__())
    return IOCLoss

if __name__ == "__main__":
    n_steps, n_episodes = 200, 10
    env_id = "IP_custom-v2"
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    rwfn = RewfromMat(num_obs).double()
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
                sampleL = generate_trajectories(algo.policy, env, sample_until) #sample trajectory
            sampleE = random.sample(trajectories, 5) #Expert Demo trajectory
            # todo: cal_sample_cost: calculate IOC Loss using expert and experienced trajectories and rew func.
            IOCLoss = cal_sample_cost(sampleL, sampleE, rwfn.train()) #calculate sample's cost again
            optimizer.zero_grad()
            IOCLoss.backward()
            optimizer.step()
        env = DummyVecEnv([lambda: RewardWrapper(gym.make(env_id, n_steps=n_steps), rwfn.eval())])
        algo.set_env(env)
        algo.learn(total_timesteps=100)
