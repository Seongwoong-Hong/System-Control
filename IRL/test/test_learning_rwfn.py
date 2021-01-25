import torch, gym, gym_envs, pickle, os, shutil

from stable_baselines3.common.vec_env import DummyVecEnv
from common.rollouts import get_trajectories_probs
from algo.torch.sac import SAC
from algo.torch.ppo import PPO, MlpPolicy
from common.modules import NNCost
from imitation.data import rollout
from common.wrappers import CostWrapper

if __name__ == "__main__":
    n_steps, n_episodes = 100, 10
    env_id = "IP_custom-v1"
    expert_dir = os.path.join("..", "demos", "expert_bar_100.pkl")
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    inp = num_obs+num_act
    costfn = NNCost(arch=[num_obs], device='cpu').double().to('cpu')
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
        for k in range(1):
            with torch.no_grad():
                learner_trajs += rollout.generate_trajectories(algo.policy, env, sample_until)
                expert_trans = get_trajectories_probs(expert_trajs, algo.policy)
                learner_trans = get_trajectories_probs(learner_trajs, algo.policy)
            costfn.sample_trajectory_sets(learner_trans, expert_trans)
            costfn.learn(epoch=1)

        env = DummyVecEnv([lambda: CostWrapper(gym.make(env_id, n_steps=n_steps), costfn._eval())])
        algo.set_env(env)
        algo.learn(total_timesteps=204800)
