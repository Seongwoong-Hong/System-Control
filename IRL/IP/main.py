import torch, gym, gym_envs, pickle, os, sys
from stable_baselines3.common.vec_env import DummyVecEnv
from common.callbacks import VideoRecorderCallback
from common.rollouts import get_trajectories_probs
from algo.torch.ppo import PPO
from algo.torch.IRL import RewfromMat
from imitation.data import rollout
from mujoco_py import GlfwContext

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, rwfn):
        super(RewardWrapper, self).__init__(env)
        self.rwfn = rwfn

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation), done, info

    def reward(self, observation):
        return self.rwfn.forward(torch.from_numpy(observation).to(rwfn.device))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        device = 'cuda:0'
    else:
        device = sys.argv[1]
    n_steps, n_episodes = 40, 10
    env_id = "IP_custom-v2"
    log_dir = os.path.join(os.path.dirname(__file__), "tmp", "log")
    model_dir = os.path.join(os.path.dirname(__file__), "tmp", "model")
    expert_dir = os.path.join(os.path.dirname(__file__), "demos", "expert_20.pkl")
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    rwfn = RewfromMat(num_obs, device=device).double().to(device)
    env = DummyVecEnv([lambda: env])
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    print("Start Guided Cost Learning...  Using {} environment".format(env_id))
    GlfwContext(offscreen=True)
    algo = PPO("MlpPolicy",
               env=env,
               n_steps=2048,
               batch_size=128,
               gamma=0.99,
               gae_lambda=0.95,
               ent_coef=0.01,
               verbose=1,
               device=device,
               tensorboard_log=log_dir)
    video_recorder = VideoRecorderCallback(log_dir+"/video",
                                           gym.make(env_id, n_steps=n_steps),
                                           n_eval_episodes=5,
                                           render_freq=512000)
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
        learner_trajs = []
    for _ in range(10):
        # update cost function
        for k in range(20):
            with torch.no_grad():
                learner_trajs += rollout.generate_trajectories(algo.policy, env, sample_until)
                expert_trans = get_trajectories_probs(expert_trajs, algo.policy)
                learner_trans = get_trajectories_probs(learner_trajs, algo.policy)
            rwfn.sample_trajectory_sets(learner_trans, expert_trans)
            rwfn.learn(epoch=50)

        # update policy
        env = DummyVecEnv([lambda: RewardWrapper(gym.make(env_id, n_steps=n_steps), rwfn._eval())])
        algo.set_env(env)
        algo.learn(total_timesteps=1024000, callback=video_recorder)
    torch.save(rwfn, model_dir+"/rwfn.pt")
    algo.save(model_dir+"/ppo")