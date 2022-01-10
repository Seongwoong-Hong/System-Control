import os
import pickle
import pytest
import torch as th
import numpy as np

from common.util import make_env
from common.verification import CostMap
from common.wrappers import *
from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from IRL.scripts.project_policies import def_policy

from imitation.algorithms import bc
from matplotlib import pyplot as plt
from matplotlib import cm


@pytest.fixture()
def irl_path():
    return os.path.abspath(os.path.join("..", "..", "IRL"))


def test_draw_costmap():
    inputs = [[[0, 1, 2, 3], [3, 4, 5, 6]]]
    fig = CostMap.draw_costmap(inputs)


def test_cal_cost(irl_path):
    from imitation.data.rollout import flatten_trajectories
    expert_dir = os.path.join(irl_path, "demos", "HPC", "lqrTest.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    transitions = []
    for traj in expert_trajs:
        transitions += [flatten_trajectories([traj])]
    cost_dir = os.path.join(irl_path, "tmp", "log", "HPC", "ppo", "AIRL_test", "69", "model", "discrim.pkl")
    with open(cost_dir, "rb") as f:
        disc = pickle.load(f)
    reward_fn = disc.reward_net.base_reward_net.double()
    orders = [i for i in range(len(expert_trajs))]
    agent = {'transitions': transitions, 'cost_fn': reward_fn, 'orders': orders}
    inputs = CostMap.cal_cost([agent])
    CostMap.draw_costmap(inputs)


def test_process_agent(tenv, irl_path):
    cost_dir = os.path.join(irl_path, "tmp", "log", "HPC", "ppo", "AIRL_test", "69", "model")
    with open(cost_dir + "/discrim.pkl", "rb") as f:
        disc = pickle.load(f)
    cost_fn = disc.reward_net.base_reward_net.double()
    algo = PPO.load(cost_dir + "/gen.zip")
    agent = {"algo": algo, "env": tenv, "cost_fn": cost_fn}
    agents = CostMap.process_agent(agent)
    inputs = CostMap.cal_cost(agents)
    CostMap.draw_costmap(inputs)


def test_costmap(tenv, irl_path):
    reward_dir = os.path.join(irl_path, "tmp", "log", "HPC", "ppo", "AIRL_test", "69", "model")
    with open(reward_dir + "/discrim.pkl", "rb") as f:
        disc = pickle.load(f).double()
    reward_fn = disc.reward_net.base_reward_net
    agent = PPO.load(reward_dir + "/gen.zip")
    # expt = def_policy("HPC", tenv)
    cost_map = CostMap(reward_fn, tenv, agent)


def test_expt_reward(irl_path):
    env_type = "IDP"
    name = "IDP_custom"
    rewards = []
    env = make_env(f"{env_type}_custom-v1", use_vec_env=False)
    load_dir = irl_path + f"/tmp/log/{name}/BC/sq_lqr_ppo_ppoagent_noreset/model/000/agent"
    expt = def_policy("IDP", env)
    agent = PPO.load(load_dir)
    for i in range(10):
        done = False
        reward = 0
        obs = env.reset()
        while not done:
            act, _ = expt.predict(obs, deterministic=True)
            obs, rew, done, _ = env.step(act)
            reward += rew.item()
        rewards.append(reward)
    print(np.mean(rewards))
    plt.plot(rewards)
    plt.show()


def test_agent_reward(irl_path, subj="sub06", actu=1):
    import time
    from scipy import io
    from common.rollouts import generate_trajectories_without_shuffle
    from algos.tabular.viter import FiniteSoftQiter, SoftQiter
    from imitation.data.rollout import make_sample_until, flatten_trajectories

    rendering = False
    plotting = True

    def feature_fn(x):
        # return x
        # return x ** 2
        return th.cat([x, x ** 2], dim=1)

    env_type = "2DTarget_disc"
    name = f"{env_type}"
    expt = f"50/softqiter_more_random"
    load_dir = f"{irl_path}/tmp/log/{env_type}/MaxEntIRL/ext_{expt}_finite2/model"
    with open(load_dir + "/reward_net.pkl", "rb") as f:
        reward_fn = pickle.load(f).cpu()
    bsp = io.loadmat(os.path.join(irl_path, "demos", "HPC", subj, subj + "i1.mat"))['bsp']
    with open(f"{irl_path}/demos/{env_type}/{expt}.pkl", "rb") as f:
        expert_traj = pickle.load(f)
    init_states = []
    for traj in expert_traj:
        init_states.append(traj.obs[0])
    reward_fn.feature_fn = feature_fn
    env = make_env(f"{name}-v2", num_envs=1, wrapper=RewardWrapper, wrapper_kwrags={'rwfn': reward_fn.eval()})
    # env = make_env(f"{name}-v2", num_envs=1, N=[9, 19, 19, 27], bsp=bsp,
    #                wrapper=ActionRewardWrapper, wrapper_kwrags={'rwfn': reward_fn.eval()})
    learned_agent = FiniteSoftQiter(env, gamma=1, alpha=0.01, device='cpu', verbose=False)
    learned_agent.learn(0)
    agent = SoftQiter(env)
    agent.policy.policy_table = learned_agent.policy.policy_table[0]
    # eval_env = make_env(f"{name}-v0", num_envs=1, N=[9, 19, 19, 27], bsp=bsp, init_states=init_states)
    eval_env = make_env(f"{name}-v0", num_envs=1, init_states=init_states)
    agent.set_env(eval_env)
    sample_until = make_sample_until(n_timesteps=None, n_episodes=len(expert_traj))
    agent_traj = generate_trajectories_without_shuffle(
        agent, eval_env, sample_until, deterministic_policy=False)

    expt_obs = flatten_trajectories(expert_traj).obs
    expt_acts = flatten_trajectories(expert_traj).acts
    agent_obs = flatten_trajectories(agent_traj).obs
    agent_acts = flatten_trajectories(agent_traj).acts

    print(f"Mean obs difference ({subj}_{actu}): {np.abs(expt_obs - agent_obs).mean()}")
    print(f"Mean acts difference ({subj}_{actu}): {np.abs(expt_acts - agent_acts).mean()}")

    if plotting:
        x_value = range(1, 201)
        obs_fig = plt.figure(figsize=[18, 12], dpi=150.0)
        acts_fig = plt.figure(figsize=[18, 12], dpi=150.0)
        for ob_idx in range(2):
            ax = obs_fig.add_subplot(2, 1, ob_idx + 1)
            for traj_idx in range(len(expert_traj)):
                ax.plot(x_value, agent_traj[traj_idx].obs[:-1, ob_idx], color='k')
                ax.plot(x_value, expert_traj[traj_idx].obs[:-1, ob_idx], color='b')
        for act_idx in range(2):
            ax = acts_fig.add_subplot(2, 1, act_idx + 1)
            for traj_idx in range(len(expert_traj)):
                ax.plot(x_value, agent_traj[traj_idx].acts[:, act_idx], color='k')
                ax.plot(x_value, expert_traj[traj_idx].acts[:, act_idx], color='b')
        obs_fig.tight_layout()
        acts_fig.tight_layout()
        plt.show()

    if rendering:
        for i in range(10):
            done = False
            obs = eval_env.reset()
            while not done:
                act, _ = agent.predict(obs, deterministic=True)
                obs, rew, done, _ = eval_env.step(act)
                eval_env.render()
                time.sleep(env.get_attr("dt")[0])


def test_learned_results(irl_path):
    for subj in [f"sub{i:02d}" for i in [1, 4, 6]]:
        for actu in range(1, 7):
            test_agent_reward(irl_path, subj, actu)


def test_learned_reward_mat(irl_path):
    from matplotlib import cm

    def feature_fn(x):
        return x
        # return x ** 2
        # return th.cat([x, x**2], dim=1)

    def normalized(x):
        return (x - x.min()) / (x.max() - x.min())

    expt_env = make_env("2DTarget_disc-v2")
    with open(irl_path + f"/tmp/log/2DTarget_disc/MaxEntIRL/cnn_50/softqiter_more_random_finite2/model/reward_net.pkl",
              "rb") as f:
        rwfn = pickle.load(f)
    rwfn.feature_fn = feature_fn
    agent_env = make_env("2DTarget_disc-v2", wrapper=RewardWrapper, wrapper_kwrags={'rwfn': rwfn.eval()})
    expt_reward_mat = normalized(expt_env.get_reward_mat())
    agent_reward_mat = normalized(agent_env.get_reward_mat())
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(np.vstack([expt_reward_mat[:, 500 * i:500 * (i + 1)] for i in range(5)]), cmap=cm.rainbow)
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(np.vstack([agent_reward_mat[:, 500 * i:500 * (i + 1)] for i in range(5)]), cmap=cm.rainbow)
    plt.show()
    print(np.abs((expt_reward_mat - agent_reward_mat).mean()))
