import gym
import gym_envs
import os
import pickle
import torch
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv

from algo.torch.ppo import PPO, MlpPolicy
from common.modules import CostNet
from common.rollouts import get_trajectories_probs
from common.wrappers import CostWrapper

if __name__ == "__main__":
    n_steps, n_episodes = 100, 10
    env_id = "IDP_custom-v0"
    expert_dir = os.path.join("..", "demos", "IDP", "expert.pkl")
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    inp = num_obs+num_act
    costfn = CostNet(arch=[num_obs], device='cpu', num_act=num_act).double().to('cpu')
    env = DummyVecEnv([lambda: CostWrapper(env, costfn)])
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    print("Start Guided Cost Learning...  Using {} environment".format(env_id))
    algo = PPO(MlpPolicy,
               env=env,
               verbose=1,
               device='cpu')

    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
        learner_trajs = []
    for _ in range(20):
        with torch.no_grad():
            learner_trajs += rollout.generate_trajectories(algo.policy, env, sample_until)
            expert_trans = get_trajectories_probs(expert_trajs, algo.policy)
            learner_trans = get_trajectories_probs(learner_trajs, algo.policy)
        for k in range(1):
            costfn.sample_trajectory_sets(learner_trans, expert_trans)
            costfn.learn(epoch=1)

        env = DummyVecEnv([lambda: CostWrapper(gym.make(env_id, n_steps=n_steps), costfn.eval_())])
        algo.set_env(env)
        algo.learn(total_timesteps=2048)
