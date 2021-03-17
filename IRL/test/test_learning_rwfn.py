import gym_envs
import os
import pickle
import torch
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from algo.torch.ppo import PPO
from IRL.project_policies import def_policy
from common.modules import CostNet
from common.rollouts import get_trajectories_probs
from common.wrappers import CostWrapper
from scipy import io

if __name__ == "__main__":
    n_steps, n_episodes = 300, 5
    env_type = "HPC"
    algo_type = "ppo"
    name = "{}/{}/2021-2-14-15-18-36".format(env_type, algo_type)
    num = 2
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    # costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
    # algo = PPO.load(model_dir + "/ppo{}.zip".format(num), device=costfn.device)
    env_id = "HPC_custom-v0"
    expert_dir = os.path.join("..", "demos", "HPC", "sub01.pkl")
    sub = "sub01"
    pltqs = []
    for i in range(35):
        file = "../demos/HPC/" + sub + "/" + sub + "i%d.mat" % 1
        pltqs += [io.loadmat(file)['pltq']]

    costfn = CostNet(arch=[4], device='cpu', num_act=2, verbose=1, num_samp=5).double().to('cpu')
    env = DummyVecEnv([lambda: CostWrapper(gym_envs.make(env_id, n_steps=n_steps, pltqs=pltqs), costfn) for i in range(5)])
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    inp = num_obs+num_act

    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    print("Start Guided Cost Learning...  Using {} environment".format(env_id))
    algo = def_policy("ppo", env, device='cpu', log_dir='./', verbose=1)

    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
        learner_trajs = []

    for _ in range(20):
        # Add sample trajectories from current policy
        learner_trajs = []
        with torch.no_grad():
            for _ in range(10):
                learner_trajs += rollout.generate_trajectories(algo.policy, env, sample_until)
            expert_trans = get_trajectories_probs(expert_trajs, algo.policy)
            learner_trans = get_trajectories_probs(learner_trajs, algo.policy)

        # update cost function
        for k in range(3):
            costfn.learn(learner_trans, expert_trans, epoch=1)
        # update policy using PPO
        env = DummyVecEnv([
            lambda: CostWrapper(gym_envs.make(env_id, n_steps=n_steps, pltqs=pltqs), costfn.eval_()) for i in range(5)])
        algo.set_env(env)
        with torch.no_grad():
            for n, param in algo.policy.named_parameters():
                if 'log_std' in n:
                    param.copy_(torch.zeros(*param.shape))
        algo.learn(total_timesteps=40960, tb_log_name="log")

