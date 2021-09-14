import os
import pickle
import numpy as np
import torch as th

from copy import deepcopy
from scipy import io
from matplotlib import cm
from matplotlib import pyplot as plt
from imitation.data.rollout import flatten_trajectories, make_sample_until, generate_trajectories
from stable_baselines3.common.vec_env import DummyVecEnv

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import verify_policy
from common.wrappers import ActionWrapper
from IRL.scripts.project_policies import def_policy


def learned_cost():
    env_type = "HPC"
    env_id = f"{env_type}_custom"
    subj = "ppo"
    name = f"extcnn_{subj}_stdreset"
    proj_path = os.path.abspath(os.path.join("..", "tmp", "log", env_id, "BC", name))
    print(proj_path)
    with open(f"../demos/{env_type}/{subj}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    expt_trans = flatten_trajectories(expert_trajs)
    test_len = len(expert_trajs)
    subpath = os.path.abspath(os.path.join("..", "demos", env_type, "sub01", "sub01"))
    wrapper = ActionWrapper if env_type == "HPC" else None
    env = make_env(f"{env_id}-v0", wrapper=wrapper, subpath=subpath)
    sample_until = make_sample_until(n_timesteps=None, n_episodes=test_len)
    i = 0
    cost_list = []
    expt_input = th.from_numpy(np.concatenate([expt_trans.obs, env.action(expt_trans.acts)], axis=1)).double()
    while os.path.isdir(os.path.join(proj_path, "model", f"{i:03d}")):
        env = make_env(f"{env_id}-v0", wrapper=wrapper, subpath=subpath)
        agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"))
        if os.path.isfile(os.path.abspath(proj_path + f"/model/{i:03d}/normalization.pkl")):
            stats_path = os.path.abspath(proj_path + f"/model/{i:03d}/normalization.pkl")
            venv = make_env(f"{env_id}-v0", use_vec_env=True, num_envs=1, use_norm=stats_path, wrapper=wrapper)
            expt_input = th.from_numpy(np.concatenate([venv.normalize_obs(expt_trans.obs), expt_trans.acts], axis=1)).double()
        agent_trajs = generate_trajectories(agent, DummyVecEnv([lambda: env]), sample_until=sample_until, deterministic_policy=False)
        agent_trans = flatten_trajectories(agent_trajs)
        agent_input = th.from_numpy(np.concatenate([agent_trans.obs, env.action(agent_trans.acts)], axis=1)).double()
        with open(os.path.join(proj_path, "model", f"{i:03d}", "reward_net.pkl"), "rb") as f:
            reward_fn = pickle.load(f).double()
        agent_cost = -reward_fn(agent_input).sum().item() / test_len
        expt_cost = -reward_fn(expt_input).sum().item() / test_len
        print(f"{i:03d} Agent Cost:", agent_cost, "Expert Cost:", expt_cost)
        cost_list.append([expt_cost, agent_cost])
        i += 1
    # plt.plot(cost_list)
    # plt.savefig(f"figures/IDP/MaxEntIRL/agent_cost_each_iter/expt_{name}.png")
    # plt.show()


def expt_cost():
    def expt_fn(inp):
        return inp[:, :2].square().sum() + 0.1 * inp[:, 2:4].square().sum() + 1e-6 * (300 * inp[:, -2:]).square().sum()
    env_type = "HPC"
    env_id = f"{env_type}_pybullet"
    subj = "ppo"
    name = f"ext_{subj}_linear_noreset"
    print(name)
    proj_path = os.path.abspath(os.path.join("..", "tmp", "log", env_id, "BC", name))
    with open(f"../demos/{env_type}/{subj}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    expt_trans = flatten_trajectories(expert_trajs)
    test_len = len(expert_trajs)
    subpath = os.path.abspath(os.path.join("..", "demos", env_type, "sub01", "sub01"))
    wrapper = ActionWrapper if env_type == "HPC" else None
    venv = make_env(f"{env_id}-v0", use_vec_env=True, num_envs=1, wrapper=wrapper, subpath=subpath)
    th_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1))
    print(f"expt_cost: {expt_fn(th_input).item() / test_len}")
    sample_until = make_sample_until(n_timesteps=None, n_episodes=test_len)
    i = 0
    cost_list = []
    while os.path.isdir(os.path.join(proj_path, "model", f"{i:03d}")):
        agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"), device='cpu')
        if os.path.isfile(proj_path + f"/{i:03d}/normalization.pkl"):
            stats_path = proj_path + f"/model/{i:03d}/normalization.pkl"
            venv = make_env(f"{env_id}-v0", num_envs=1, use_norm=True,
                            wrapper=wrapper, stats_path=stats_path, subpath=subpath)
        venv.render(mode="None")
        agent_trajs = generate_trajectories(agent, venv, sample_until=sample_until, deterministic_policy=True)
        agent_trans = flatten_trajectories(agent_trajs)
        th_input = th.from_numpy(np.concatenate([agent_trans.obs, agent_trans.acts], axis=1))
        print(f"{i:03d} Cost:", expt_fn(th_input).item() / test_len)
        cost_list.append(expt_fn(th_input).item() / test_len)
        i += 1
    # plt.plot(cost_list)
    # plt.savefig(f"figures/IDP/MaxEntIRL/expt_cost_each_iter/{name}.png")
    # plt.show()
    print(f"minimum cost index: {np.argmin(cost_list)}")


def compare_obs():
    env_type = "HPC"
    env_id = f"{env_type}_custom"
    subj = "ppo"
    name = f"cnn_{subj}_noreset"
    print(name)
    proj_path = os.path.abspath(os.path.join("..", "tmp", "log", env_id, "BC", name))
    assert os.path.isdir(proj_path)
    subpath = os.path.abspath(os.path.join("..", "demos", env_type, "sub01", "sub01"))
    with open(f"../demos/{env_type}/{subj}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    wrapper = ActionWrapper if env_type == "HPC" else None
    error_list, max_list = [], []
    i = 0
    while os.path.isdir(os.path.join(proj_path, "model", f"{i:03d}")):
        agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"), device='cpu')
        stats_path = None
        if os.path.isfile(os.path.join(proj_path, "model", f"{i:03d}", "normalization.pkl")):
            stats_path = os.path.join(proj_path, "model", f"{i:03d}", "normalization.pkl")
        env = make_env(f"{env_id}-v0", num_envs=1, wrapper=wrapper, subpath=subpath, use_norm=stats_path)
        _, agent_obs, _ = verify_policy(env, agent, deterministic=True, render="None", repeat_num=35)
        if stats_path is not None:
            agent_obs = env.unnormalize_obs(agent_obs)
        errors, maximums = [], []
        for k in range(35):
            errors += [abs(expert_trajs[k].obs[:, :2] - agent_obs[k][:, :2]).mean()]
            maximums += [abs(expert_trajs[k].obs[:, :2] - agent_obs[k][:, :2]).max()]
        error_list.append(sum(errors) / len(errors))
        max_list.append(sum(maximums) / len(maximums))
        print(f"{i:03d} Error: {error_list[-1]}, Max: {max_list[-1]}")
        i += 1
    print(f"minimum error index: {np.argmin(error_list)}, minimum max index: {np.argmin(max_list)}")


def feature():
    from imitation.data.rollout import flatten_trajectories, make_sample_until, generate_trajectories
    from common.wrappers import ActionWrapper
    env_type = "HPC"
    env_id = f"{env_type}_custom"
    subj = "sub01"
    name = f"extcnn_{subj}_reset_weightnorm"
    i = 25
    print(name)
    proj_path = os.path.join("..", "tmp", "log", env_id, "BC", name)
    with open(f"../demos/{env_type}/{subj}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    expt_trans = flatten_trajectories(expert_trajs)
    test_len = len(expert_trajs)
    subpath = os.path.join("..", "demos", env_type, subj)
    wrapper = ActionWrapper if env_type == "HPC" else None
    venv = make_env(f"{env_id}-v0", use_vec_env=True, num_envs=1, wrapper=wrapper, subpath=subpath + f"/{subj}")
    expt_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1))
    sample_until = make_sample_until(n_timesteps=None, n_episodes=test_len)
    agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"))
    agent_trajs = generate_trajectories(agent, venv, sample_until=sample_until, deterministic_policy=False)
    agent_trans = flatten_trajectories(agent_trajs)
    agent_input = th.from_numpy(np.concatenate([agent_trans.obs, agent_trans.acts], axis=1))
    with open(os.path.join(proj_path, "model", f"{i:03d}", "reward_net.pkl"), "rb") as f:
        reward_fn = pickle.load(f).double()
    print("env")


if __name__ == "__main__":
    def feature_fn(x):
        return x
        # return x.square()
        # return th.cat([x, x.square()], dim=1)
    compare_obs()
