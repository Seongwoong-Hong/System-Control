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


def test_learned_reward_mat(irl_path):
    from matplotlib import cm

    def feature_fn(x):
        return x
        # return x ** 2
        # return th.cat([x, x**2], dim=1)

    def normalized(x):
        return (x - x.min()) / (x.max() - x.min())

    env_name = "DiscretizedHuman"
    expt_env = make_env(f"{env_name}-v2", N=[9, 19, 19, 27])
    with open(f"{irl_path}/tmp/log/{env_name}/MaxEntIRL/cnn_09191927/sub01_1_half_finite2/model/reward_net.pkl",
              "rb") as f:
        rwfn = pickle.load(f)
    rwfn.feature_fn = feature_fn
    agent_env = make_env(f"{env_name}-v2", N=[9, 19, 19, 27], wrapper=ActionNormalizeRewardWrapper,
                         wrapper_kwrags={'rwfn': rwfn.eval()})
    expt_reward_mat = normalized(expt_env.get_reward_mat())
    agent_reward_mat = normalized(agent_env.get_reward_mat())
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(np.vstack([expt_reward_mat[:, 500 * i:500 * (i + 1)] for i in range(5)]), cmap=cm.rainbow)
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(np.vstack([agent_reward_mat[:, 500 * i:500 * (i + 1)] for i in range(5)]), cmap=cm.rainbow)
    plt.show()
    print(np.abs((expt_reward_mat - agent_reward_mat).mean()))
