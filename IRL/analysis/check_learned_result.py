import os
import pickle
import numpy as np
import torch as th

from copy import deepcopy
from scipy import io
from matplotlib import cm
from matplotlib import pyplot as plt
from imitation.data.rollout import flatten_trajectories, make_sample_until, generate_trajectories

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import verify_policy
from common.wrappers import ActionWrapper
from IRL.scripts.project_policies import def_policy


def learned_cost():
    env_type = "HPC"
    env_id = f"{env_type}_custom"
    subj = "sub01"
    name = f"extcnn_{subj}_criticreset_0.2_grad1000"
    print(name)
    proj_path = os.path.abspath(os.path.join("..", "tmp", "log", env_id, "MaxEntIRL", name))
    with open(f"../demos/{env_type}/{subj}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    expt_trans = flatten_trajectories(expert_trajs)
    test_len = len(expert_trajs)
    subpath = os.path.abspath(os.path.join("..", "demos", env_type, subj))
    wrapper = ActionWrapper if env_type == "HPC" else None
    venv = make_env(f"{env_id}-v0", use_vec_env=True, num_envs=1, wrapper=wrapper, subpath=subpath + f"/{subj}")
    th_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1))
    sample_until = make_sample_until(n_timesteps=None, n_episodes=test_len)
    i = 0
    cost_list = []
    while os.path.isdir(os.path.join(proj_path, "model", f"{i:03d}")):
        # agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"))
        # if os.path.isfile(os.path.abspath(proj_path + f"/{i:03d}/normalization.pkl")):
        #     stats_path = os.path.abspath(proj_path + f"/model/{i:03d}/normalization.pkl")
        #     venv = make_env("IDP_custom-v0", use_vec_env=True, num_envs=1, use_norm=True, stats_path=stats_path)
        # agent_trajs = generate_trajectories(agent, venv, sample_until=sample_until, deterministic_policy=False)
        # agent_trans = flatten_trajectories(agent_trajs)
        # th_input = th.from_numpy(np.concatenate([agent_trans.obs, agent_trans.acts], axis=1)).double()
        with open(os.path.join(proj_path, "model", f"{i:03d}", "reward_net.pkl"), "rb") as f:
            reward_fn = pickle.load(f).double()
        print(f"{i:03d} Cost:", -reward_fn(th_input).sum().item() / test_len)
        cost_list.append(-reward_fn(th_input).sum().item() / test_len)
        i += 1
    plt.plot(cost_list)
    # plt.savefig(f"figures/IDP/MaxEntIRL/agent_cost_each_iter/expt_{name}.png")
    plt.show()


def expt_cost():
    def expt_fn(inp):
        return inp[:, :2].square().sum() + 0.1 * inp[:, 2:4].square().sum() + 1e-5 * (200 * inp[:, -2:]).square().sum()
    env_type = "HPC"
    env_id = f"{env_type}_pybullet"
    subj = "sub01"
    name = f"extcnn_{subj}_noreset_rewfirst"
    print(name)
    proj_path = os.path.abspath(os.path.join("..", "tmp", "log", env_id, "BC", name))
    with open(f"../demos/{env_type}/{subj}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    expt_trans = flatten_trajectories(expert_trajs)
    test_len = len(expert_trajs)
    subpath = os.path.abspath(os.path.join("..", "demos", env_type, subj))
    wrapper = ActionWrapper if env_type == "HPC" else None
    venv = make_env(f"{env_id}-v0", use_vec_env=True, num_envs=1, wrapper=wrapper, subpath=subpath + f"/{subj}")
    th_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1))
    print(f"expt_cost: {expt_fn(th_input).item() / test_len}")
    sample_until = make_sample_until(n_timesteps=None, n_episodes=test_len)
    i = 0
    cost_list = []
    while os.path.isdir(os.path.join(proj_path, "model", f"{i:03d}")):
        agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"), device='cpu')
        if os.path.isfile(proj_path + f"/{i:03d}/normalization.pkl"):
            stats_path = proj_path + f"/model/{i:03d}/normalization.pkl"
            venv = make_env(f"{env_id}-v0", num_envs=1, use_norm=True, wrapper=wrapper,
                            stats_path=stats_path, subpath=subpath + f"/{subj}")
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
    subj = "sub01"
    name = f"extcnn_{subj}_noreset_weightnorm3"
    print(name)
    proj_path = os.path.abspath(os.path.join("..", "tmp", "log", env_id, "BC", name))
    subpath = os.path.abspath(os.path.join("..", "demos", env_type, subj))
    wrapper = ActionWrapper if env_type == "HPC" else None
    error_list, max_list = [], []
    i = 0
    while os.path.isdir(os.path.join(proj_path, "model", f"{i:03d}")):
        agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"), device='cpu')
        env = make_env(f"{env_id}-v0", use_vec_env=False, num_envs=1, wrapper=wrapper, subpath=subpath + f"/{subj}")
        _, agent_obs, _ = verify_policy(env, agent, deterministic=True, render="None", repeat_num=35)
        expt_obs = [io.loadmat(f"../demos/HPC/{subj}/{subj}i{i + 1}.mat")['state'] for i in range(35)]
        errors, maximums = [], []
        for k in range(35):
            errors += [abs(expt_obs[k][:, :2] - agent_obs[k][:, :2]).mean()]
            maximums += [abs(expt_obs[k][:, :2] - agent_obs[k][:, :2]).max()]
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
    name = f"extcnn_{subj}_deep_noreset_rewfirst"
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
        # return x
        # return x.square()
        return th.cat([x, x.square()], dim=1)
    compare_obs()
