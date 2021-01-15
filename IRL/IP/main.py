import torch, gym, gym_envs, pickle, os, sys
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from common.callbacks import VideoRecorderCallback
from common.rollouts import get_trajectories_probs
from common.wrappers import CostWrapper
from common.modules import NNCost
from algo.torch.ppo import PPO
from imitation.data import rollout
from mujoco_py import GlfwContext

if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SyntaxError("Please enter name for logging")
    elif len(sys.argv) == 2:
        name = sys.argv[1]
    elif len(sys.argv) == 3:
        name = sys.argv[1]
        device = sys.argv[2]
    else:
        raise SyntaxError("Too many system inputs")
    n_steps, n_episodes = 100, 10
    env_id = "IP_custom-v2"
    log_dir = os.path.join(os.path.dirname(__file__), "tmp", "log")
    model_dir = os.path.join(os.path.dirname(__file__), "tmp", "model")
    expert_dir = os.path.join(os.path.dirname(__file__), "demos", "expert_bar_100.pkl")
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    costfn = NNCost(num_obs+num_act, device=device).double().to(device)
    env = DummyVecEnv([lambda: env])
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    print("Start Guided Cost Learning...  Using {} environment.\nThe Name for logging is {}".format(env_id, name))
    GlfwContext(offscreen=True)
    algo = PPO("MlpPolicy",
               env=env,
               n_steps=2048,
               batch_size=128,
               gamma=0.99,
               gae_lambda=0.95,
               ent_coef=0.01,
               verbose=0,
               device=device,
               tensorboard_log=log_dir)
    video_recorder = VideoRecorderCallback(log_dir+"/video/bar_2",
                                           gym.make(env_id, n_steps=n_steps),
                                           n_eval_episodes=5,
                                           render_freq=102400)
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
            costfn.sample_trajectory_sets(learner_trans, expert_trans)
            costfn.learn(epoch=50)

        # update policy
        env = DummyVecEnv([lambda: CostWrapper(gym.make(env_id, n_steps=n_steps), costfn._eval())])
        algo.set_env(env)
        algo.learn(total_timesteps=512000, callback=video_recorder, tb_log_name=name)
    torch.save(costfn, model_dir+"/"+name+"_costfn.pt")
    algo.save(model_dir+"/"+name+"_ppo")