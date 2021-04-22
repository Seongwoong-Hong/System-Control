import datetime
import gym_envs
import os
import pickle
import torch
from imitation.data import rollout
from mujoco_py import GlfwContext
from stable_baselines3.common.vec_env import DummyVecEnv

from IRL.project_policies import def_policy
from common.modules import CostNet
from common.rollouts import get_trajectories_probs
from common.wrappers import CostWrapper

if __name__ == "__main__":
    env_type = 'IDP'
    device = 'cpu'
    algo_type = 'ppo'
    now = datetime.datetime.now()

    expert_dir = os.path.join("..", "demos", env_type, "expert_human.pkl")

    n_steps, n_episodes = 900, 5
    steps_for_learn = 3072000
    env_id = "{}_human-v0".format(env_type)
    env = gym_envs.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]

    costfn = CostNet(arch=[num_obs, 2*num_obs],
                     act_fcn=torch.nn.ReLU,
                     device=device,
                     num_expert=15,
                     num_samp=n_episodes,
                     lr=3e-4,
                     decay_coeff=0.0,
                     num_act=num_act
                     ).double().to(device)

    env = DummyVecEnv([lambda: CostWrapper(env, costfn) for i in range(5)])
    GlfwContext(offscreen=True)

    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
        learner_trajs = []

    algo = def_policy(algo_type, env, device=device, log_dir="./")

    for i in range(30):
        # Add sample trajectories from current policy
        learner_trajs = []
        env = DummyVecEnv([lambda: CostWrapper(gym_envs.make(env_id, n_steps=n_steps), costfn)])
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
            lambda: CostWrapper(gym_envs.make(env_id, n_steps=n_steps), costfn.eval_()) for i in range(5)])
        algo.set_env(env)
        with torch.no_grad():
            for n, param in algo.policy.named_parameters():
                if 'log_std' in n:
                    param.copy_(torch.zeros(*param.shape))
        algo.learn(total_timesteps=40960, tb_log_name="log")
