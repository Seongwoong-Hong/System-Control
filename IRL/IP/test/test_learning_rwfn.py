import torch, gym, gym_envs, pickle, os

from stable_baselines3.common.vec_env import DummyVecEnv
from common.rollouts import get_trajectories_probs
from algo.torch.ppo import PPO
from algo.torch.IRL import RewfromMat
from imitation.data import rollout
from common.wrappers import ActionRewardWrapper

if __name__ == "__main__":
    n_steps, n_episodes = 40, 10
    env_id = "IP_custom-v2"
    expert_dir = os.path.join("..", "demos", "expert_bar_aw_40.pkl")
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    rwfn = RewfromMat(num_obs+num_act, device='cpu').double().to('cpu')
    env = DummyVecEnv([lambda: ActionRewardWrapper(env, rwfn)])
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    print("Start Guided Cost Learning...  Using {} environment".format(env_id))
    algo = PPO("MlpPolicy",
               env=env,
               n_steps=2048,
               batch_size=128,
               gamma=0.99,
               gae_lambda=0.95,
               ent_coef=0.01,
               verbose=1,
               device='cpu')

    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
        learner_trajs = []

    for k in range(20):
        with torch.no_grad():
            learner_trajs += rollout.generate_trajectories(algo.policy, env, sample_until)
            expert_trans = get_trajectories_probs(expert_trajs, algo.policy)
            learner_trans = get_trajectories_probs(learner_trajs, algo.policy)
        rwfn.sample_trajectory_sets(learner_trans, expert_trans)
        rwfn.learn(epoch=50)