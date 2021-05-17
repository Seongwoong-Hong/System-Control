import os
import pickle

from common.util import make_env
from common.verification import CostMap
from algo.torch.ppo import PPO
from algo.torch.sac import SAC


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
    env_type = "IP"
    env = make_env(f"{env_type}_custom-v2", use_vec_env=False)
    load_dir = f"../tmp/log/{env_type}/MaxEntIRL/{env_type}_custom/0/model/agent.zip"
    agent = SAC.load(load_dir)
    done = False
    reward = 0
    obs = env.reset()
    while not done:
        act, _ = agent.predict(obs, deterministic=True)
        obs, rew, done, _ = env.step(act)
        reward += rew
    print(reward)


def test_agent_reward():
    from common.wrappers import RewardWrapper
    env_type = "IP"
    name = "IP_custom"
    env = make_env(f"{name}-v2", use_vec_env=False)
    load_dir = f"../tmp/log/{env_type}/MaxEntIRL/{name}/1/model"
    # agent = SAC.load(load_dir + "/agent")
    expt = PPO.load(f"../../RL/{env_type}/tmp/log/{name}/ppo/policies_2/model.pkl")
    with open(load_dir + "/reward_net.pkl", "rb") as f:
        reward_fn = pickle.load(f).double()
    env = RewardWrapper(env, reward_fn)
    done = False
    reward = 0
    obs = env.reset()
    while not done:
        act, _ = expt.predict(obs, deterministic=True)
        obs, rew, done, _ = env.step(act)
        reward += rew
    print(reward)