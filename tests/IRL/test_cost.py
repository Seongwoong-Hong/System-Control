import os
import pickle

from common.util import make_env
from common.verification import CostMap
from algos.torch.ppo import PPO
from algos.torch.sac import SAC

from matplotlib import pyplot as plt


def test_draw_costmap():
    inputs = [[[0, 1, 2, 3], [3, 4, 5, 6]]]
    fig = CostMap.draw_costmap(inputs)


def test_cal_cost():
    from imitation.data.rollout import flatten_trajectories
    expert_dir = os.path.join("..", "demos", "HPC", "lqrTest.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    transitions = []
    for traj in expert_trajs:
        transitions += [flatten_trajectories([traj])]
    cost_dir = os.path.join("..", "tmp", "log", "HPC", "ppo", "AIRL_test", "69", "model", "discrim.pkl")
    with open(cost_dir, "rb") as f:
        disc = pickle.load(f)
    reward_fn = disc.reward_net.base_reward_net.double()
    orders = [i for i in range(len(expert_trajs))]
    agent = {'transitions': transitions, 'cost_fn': reward_fn, 'orders': orders}
    inputs = CostMap.cal_cost([agent])
    CostMap.draw_costmap(inputs)


def test_process_agent(tenv):
    cost_dir = os.path.join("..", "tmp", "log", "HPC", "ppo", "AIRL_test", "69", "model")
    with open(cost_dir + "/discrim.pkl", "rb") as f:
        disc = pickle.load(f)
    cost_fn = disc.reward_net.base_reward_net.double()
    algo = PPO.load(cost_dir + "/gen.zip")
    agent = {"algo": algo, "env": tenv, "cost_fn": cost_fn}
    agents = CostMap.process_agent(agent)
    inputs = CostMap.cal_cost(agents)
    CostMap.draw_costmap(inputs)


def test_costmap(tenv):
    reward_dir = os.path.join("..", "tmp", "log", "HPC", "ppo", "AIRL_test", "69", "model")
    with open(reward_dir + "/discrim.pkl", "rb") as f:
        disc = pickle.load(f).double()
    reward_fn = disc.reward_net.base_reward_net
    agent = PPO.load(reward_dir + "/gen.zip")
    # expt = def_policy("HPC", tenv)
    cost_map = CostMap(reward_fn, tenv, agent)


def test_expt_reward():
    env_type = "IDP"
    rewards = []
    for i in range(10):
        env = make_env(f"{env_type}_custom-v2", use_vec_env=False)
        load_dir = f"../tmp/log/{env_type}/MaxEntIRL/{env_type}_custom_test1/model/{i:03d}/agent.zip"
        agent = SAC.load(load_dir)
        expt = PPO.load(f"../../RL/{env_type}/tmp/log/{env_type}_custom/ppo/policies_1/ppo0")
        done = False
        reward = 0
        obs = env.reset()
        while not done:
            act, _ = agent.predict(obs, deterministic=True)
            obs, rew, done, _ = env.step(act)
            reward += rew.item()
        rewards.append(reward)
    print(rewards)
    plt.plot(rewards)
    plt.show()


def test_agent_reward():
    from common.wrappers import RewardWrapper
    env_type = "IDP"
    name = "IDP_custom"
    rewards = []
    for i in range(10):
        env = make_env(f"{name}-v2", use_vec_env=False)
        load_dir = f"../tmp/log/{env_type}/MaxEntIRL/{name}_test1/model/{i:03d}"
        agent = SAC.load(load_dir + "/agent")
        expt = PPO.load(f"../../RL/{env_type}/tmp/log/{name}/ppo/policies_1/ppo0")
        with open(load_dir + "/reward_net.pkl", "rb") as f:
            reward_fn = pickle.load(f).double()
        env = RewardWrapper(env, reward_fn.eval())
        done = False
        reward = 0
        obs = env.reset()
        while not done:
            act, _ = agent.predict(obs, deterministic=True)
            obs, rew, done, _ = env.step(act)
            reward += rew.item()
        rewards.append(reward)
    print(rewards)
    plt.plot(rewards)
    plt.show()


def feature_fn(x):
    return x


def test_learned_cost():
    from imitation.data.rollout import flatten_trajectories
    import torch as th
    import numpy as np
    proj_path = os.path.abspath(os.path.join("..", "..", "IRL", "tmp", "log", "IDP", "MaxEntIRL", "no_lqr"))
    with open("../../IRL/demos/IDP/lqr.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    expt_trans = flatten_trajectories(expert_trajs)
    th_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1))
    i = 1
    while os.path.isdir(os.path.join(proj_path, "model", f"{i:03d}")):
        agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"))
        with open(os.path.join(proj_path, "model", f"{i:03d}", "reward_net.pkl"), "rb") as f:
            reward_fn = pickle.load(f).double()
        print(-reward_fn(th_input).mean().item() * 600)
        i += 1
