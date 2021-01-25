import torch, gym, gym_envs, pickle, os, sys, shutil, datetime
from stable_baselines3.common.vec_env import DummyVecEnv
from common.callbacks import VFCustomCallback, VideoCallback
from common.rollouts import get_trajectories_probs
from common.wrappers import CostWrapper
from common.modules import NNCost
from algo.torch.ppo import PPO, MlpPolicy
from algo.torch.sac import SAC
from imitation.data import rollout
from mujoco_py import GlfwContext

if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SyntaxError("Please enter the type of environment you use")
    elif len(sys.argv) == 2:
        type = sys.argv[1]
        device = 'cpu'
        now = datetime.datetime.now()
        name = type + "/%s-%s-%s-%s-%s-%s" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    elif len(sys.argv) == 3:
        device = sys.argv[2]
        type = sys.argv[1]
        now = datetime.datetime.now()
        name = type + "/%s-%s-%s-%s-%s-%s" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    else:
        raise SyntaxError("Too many system inputs")
    current_path = os.path.dirname(__file__)

    log_dir = os.path.join(current_path, "tmp", "log", name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    model_dir = os.path.join(log_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    ## Copy used file to logging folder
    shutil.copy(os.path.abspath(current_path + "/../common/modules.py"), model_dir)
    shutil.copy(os.path.abspath(current_path + "/../gym_envs/envs/{}_custom_exp.py".format(type)), model_dir)
    shutil.copy(os.path.abspath(__file__), model_dir)

    expert_dir = os.path.join(current_path, "demos", type, "expert.pkl")

    n_steps, n_episodes = 100, 10
    env_id = "{}_custom-v0".format(type)
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    costfn = NNCost(arch=[num_obs, num_obs],
                    device=device,
                    num_expert=10,
                    num_samp=n_episodes,
                    lr=3e-4,
                    decay_coeff=0.01,
                    num_act=num_act
                    ).double().to(device)
    env = DummyVecEnv([lambda: CostWrapper(env, costfn)])
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    print("Start Guided Cost Learning...  Using {} environment.\nThe Name for logging is {}".format(env_id, name))
    GlfwContext(offscreen=True)

    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
        learner_trajs = []

    algo = PPO(MlpPolicy,
               env=env,
               n_steps=4096,
               batch_size=128,
               gamma=0.99,
               gae_lambda=0.95,
               ent_coef=0.03,
               verbose=0,
               device=device,
               tensorboard_log=log_dir)

    video_recorder = VFCustomCallback(log_dir+"/video/"+name,
                                      gym.make(env_id, n_steps=n_steps),
                                      n_eval_episodes=5,
                                      render_freq=409600,
                                      costfn=costfn)

    for i in range(20):
        # remove old tensorboard log
        if i > 4:
            shutil.rmtree(os.path.join(log_dir, "log") + "_%d" % (i - 4))

        # Add sample trajectories from current policy
        with torch.no_grad():
            learner_trajs += rollout.generate_trajectories(algo.policy, env, sample_until)
            expert_trans = get_trajectories_probs(expert_trajs, algo.policy)
            learner_trans = get_trajectories_probs(learner_trajs, algo.policy)

        # update cost function
        start = datetime.datetime.now()
        for k in range(40):
            costfn.sample_trajectory_sets(learner_trans, expert_trans)
            costfn.learn(epoch=10)
        delta = datetime.datetime.now() - start
        print("Cost Optimization Takes {}. Now start {}th policy optimization...".format(str(delta), i+1))
        video_recorder._set_costfn(costfn=costfn)

        # update policy using PPO
        env = DummyVecEnv([lambda: CostWrapper(gym.make(env_id, n_steps=n_steps), costfn._eval())])
        algo.set_env(env)
        algo.learn(total_timesteps=409600, callback=video_recorder, tb_log_name="log")

        # save updated policy
        if (i + 1) % 2 == 0:
            torch.save(costfn, model_dir + "/costfn{}.pt".format(i + 1))
            algo.save(model_dir + "/ppo{}".format(i + 1))