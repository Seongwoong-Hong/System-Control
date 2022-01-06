import os
import pickle
import pytest
import torch as th
import numpy as np

from common.util import make_env
from common.verification import CostMap
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


def test_agent_reward(irl_path):
    import time
    from scipy import io
    from common.wrappers import ActionRewardWrapper
    from algos.tabular.viter import FiniteSoftQiter, SoftQiter

    def feature_fn(x):
        return x ** 2

    env_type = "DiscretizedHuman"
    name = f"{env_type}"
    subj = "sub01"
    load_dir = f"{irl_path}/tmp/log/ray_result/{env_type}_sq_09191927_trial/{subj}_4_3/model/000"
    with open(load_dir + "/reward_net.pkl", "rb") as f:
        reward_fn = pickle.load(f).cpu()
    bsp = io.loadmat(os.path.join(irl_path, "demos", "HPC", subj, subj + "i1.mat"))['bsp']
    with open(f"{irl_path}/demos/{env_type}/09191927/{subj}_4_3.pkl", "rb") as f:
        expert_traj = pickle.load(f)
    init_states = []
    for traj in expert_traj:
        init_states.append(traj.obs[0])
    reward_fn.feature_fn = feature_fn
    env = make_env(f"{name}-v2", num_envs=1, N=[9, 19, 19, 27], bsp=bsp,
                   wrapper=ActionRewardWrapper, wrapper_kwrags={'rwfn': reward_fn.eval()})
    learned_agent = FiniteSoftQiter(env, gamma=1, alpha=0.01, device='cpu')
    learned_agent.learn(0)
    agent = SoftQiter(env)
    agent.policy.policy_table = learned_agent.policy.policy_table[0]
    eval_env = make_env(f"{name}-v0", num_envs=1, N=[9, 19, 19, 27], bsp=bsp, init_states=init_states)
    agent.set_env(eval_env)
    for i in range(10):
        done = False
        obs = eval_env.reset()
        while not done:
            act, _ = agent.predict(obs, deterministic=True)
            obs, rew, done, _ = eval_env.step(act)
            eval_env.render()
            time.sleep(env.get_attr("dt")[0])
