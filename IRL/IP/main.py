import torch, gym, gym_envs, pickle, os, sys, shutil
from stable_baselines3.common.vec_env import DummyVecEnv
from common.callbacks import VideoRecorderCallback
from common.rollouts import get_trajectories_probs
from common.wrappers import CostWrapper
from common.modules import NNCost
from algo.torch.ppo import PPO
from algo.torch.sac import SAC
from imitation.data import rollout
from mujoco_py import GlfwContext

if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SyntaxError("Please enter name for logging")
    elif len(sys.argv) == 2:
        name = sys.argv[1]
        device = 'cuda:0'
    elif len(sys.argv) == 3:
        name = sys.argv[1]
        device = sys.argv[2]
    else:
        raise SyntaxError("Too many system inputs")
    current_path = os.path.dirname(__file__)

    log_dir = os.path.join(current_path, "tmp", "log", name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    ## Copy used file to logging folder
    shutil.copy(os.path.abspath(current_path + "/../../common/modules.py"), log_dir)
    shutil.copy(os.path.abspath(current_path + "/../../gym_envs/envs/IP_custom_cont.py"), log_dir)
    shutil.copy(os.path.abspath(__file__), log_dir)

    model_dir = os.path.join(current_path, "tmp", "model", name)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    expert_dir = os.path.join(current_path, "demos", "expert_bar_100.pkl")

    n_steps, n_episodes = 100, 10
    env_id = "IP_custom-v1"
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    inp = num_obs+num_act
    costfn = NNCost(arch=[num_obs], device=device, num_expert=5, num_samp=n_episodes, lr=1e-4).double().to(device)
    env = DummyVecEnv([lambda: CostWrapper(env, costfn)])
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    print("Start Guided Cost Learning...  Using {} environment.\nThe Name for logging is {}".format(env_id, name))
    GlfwContext(offscreen=True)

    algo = PPO("MlpPolicy",
               env=env,
               n_steps=4096,
               batch_size=128,
               gamma=0.99,
               gae_lambda=0.95,
               ent_coef=0.1,
               verbose=0,
               device=device,
               tensorboard_log=log_dir)

    video_recorder = VideoRecorderCallback(log_dir+"/video/"+name,
                                           gym.make(env_id, n_steps=n_steps),
                                           n_eval_episodes=5,
                                           render_freq=204800)
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
        learner_trajs = []
    for i in range(50):
        if i > 4:
            shutil.rmtree(os.path.join(log_dir, name) + "_%d" % (i - 4))
        with torch.no_grad():
            learner_trajs += rollout.generate_trajectories(algo.policy, env, sample_until)
            expert_trans = get_trajectories_probs(expert_trajs, algo.policy)
            learner_trans = get_trajectories_probs(learner_trajs, algo.policy)
        # update cost function
        for k in range(50):
            costfn.sample_trajectory_sets(learner_trans, expert_trans)
            costfn.learn(epoch=10)
        print("Now start policy optimization...")
        # update policy using PPO
        env = DummyVecEnv([lambda: CostWrapper(gym.make(env_id, n_steps=n_steps), costfn._eval())])
        algo.set_env(env)
        algo.learn(total_timesteps=204800, callback=video_recorder, tb_log_name=name)
        if (i+1) % 5 == 0:
            torch.save(costfn, model_dir+"/costfn{}.pt".format(i+1))
            algo.save(model_dir+"/sac{}".format(i+1))