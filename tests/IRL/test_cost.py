import os
import pickle
import pytest
import torch as th
import numpy as np

from scipy import io
from common.util import make_env, CPU_Unpickler
from common.verification import CostMap
from common.wrappers import *
from algos.tabular.viter import FiniteSoftQiter
from algos.torch.ppo import PPO
from algos.torch.sac import SAC

from imitation.algorithms import bc
from matplotlib import pyplot as plt
from matplotlib import cm


irl_path = os.path.abspath(os.path.join("..", "..", "IRL"))
def mapping(x: th.tensor):
    x_max = th.max(x)
    x_min = th.min(x)
    x_mid = (x_max + x_min) / 2
    return 2 * (x - x_mid) / (x_max - x_min)

@pytest.mark.parametrize("trial", [1, 2, 3, 4])
def test_weight_consistency(trial):
    log_dir = f"{irl_path}/tmp/log/DiscretizedHuman/MaxEntIRL/ext_01alpha_19191919_quadcost_disc/sub05_4_0001alpha_{trial}/model"
    with open(f"{log_dir}/reward_net.pkl", "rb") as f:
        weights = CPU_Unpickler(f).load().layers[-1].weight.detach()[0]
    print("\n", weights)
    # print(weights[0]/0.4, weights[1]/2, weights[2]/10, weights[3]/0.16, weights[4]/4, weights[5]/100)
    # print(weights[0] * 0.4 / weights[3], weights[1]*2/weights[4], weights[2]*10/weights[5])
    # print(weights[0]/0.2, weights[1]/1.2, weights[2]/8, weights[3]/8, weights[4]/0.04, weights[5]/1.44, weights[6]/64, weights[7]/64)
    print(weights[0]/0.05, weights[1]/0.2, weights[2]/0.3, weights[3]/0.4, weights[4]/40, weights[5]/30,
          weights[6]/(0.05**2), weights[7]/(0.2**2), weights[8]/(0.3**2), weights[9]/(0.4**2), weights[10]/1600, weights[11]/900)
    # print(weights[0], weights[1], weights[2], weights[3], weights[4], weights[5],
    #       weights[6], weights[7], weights[8], weights[9], weights[10], weights[11])

@pytest.mark.parametrize("trial", [1, 2, 3, 4])
def test_total_reward_fn_for_sqmany(trial):
    log_dir = f"{irl_path}/tmp/log/2DTarget_disc/MaxEntIRL/sqmany_001alpha_50/1alpha_{trial}/model"
    with open(f"{log_dir}/reward_net.pkl", "rb") as f:
        rwfn = CPU_Unpickler(f).load()
    weight = rwfn.layers[0].weight.detach().flatten()
    x1_idx = [0] + [4 + 4 * i for i in range(6)] + [4 + 4 * i for i in range(6)]
    x2_idx = [1] + [5 + 2 * i for i in range(49)]
    a1_idx = [2] + [102 + 2 * i for i in range(6)]
    a2_idx = [3] + [103 + 2 * i for i in range(6)]
    x1s = weight[x1_idx].sum()
    x2s = weight[x2_idx].sum()
    a1s = weight[a1_idx].sum()
    a2s = weight[a2_idx].sum()
    x1 = 0
    for i, idx in enumerate(x1_idx):
        x1 += 2 * (-i) * weight[idx]
    x2 = 0
    for i, idx in enumerate(x2_idx):
        x2 += 2 * (-i) * weight[idx]
    a1 = 0
    for i, idx in enumerate(a1_idx):
        a1 += 2 * (-i) * weight[idx]
    a2 = 0
    for i, idx in enumerate(a2_idx):
        a2 += 2 * (-i) * weight[idx]
    print(f"\nx1: {x1s}, {x1}, {x1 / x1s}")
    print(f"x2: {x2s}, {x2}, {x2 / x2s}")
    print(f"a1: {a1s}, {a1}, {a1 / a1s}")
    print(f"a2: {a2s}, {a2}, {a2 / a2s}")

def test_draw_vtable():
    def feature_fn(x):
        return th.cat([x, x**2], dim=1)
        # return th.cat([x, x**2, x**3, x**4], dim=1)

    env_name = "2DWorld_disc"
    log_dir = f"{irl_path}/tmp/log/{env_name}/MaxEntIRL/ext_01alpha_noact_20"
    end_trial = 1
    fig = plt.figure(figsize=[4.8 * (end_trial + 1), 4.8])
    ex_env = make_env(f"{env_name}-v2", num_envs=1)
    expt = FiniteSoftQiter(ex_env, gamma=1, alpha=0.2, device='cpu')
    expt.learn(0)
    ax = fig.add_subplot(1, end_trial + 1, 1)
    expt_v = mapping(expt.policy.v_table[0].cpu().reshape(10, 10))
    ex_im = ax.imshow(expt_v, interpolation='None')
    fig.colorbar(ex_im)
    for trial in range(1, end_trial + 1):
        with open(f"{log_dir}/01alpha_nobias_noact_many_{trial}/model/reward_net.pkl", "rb") as f:
            rwfn = CPU_Unpickler(f).load().to('cpu').eval()
        rwfn.feature_fn = feature_fn
        venv = make_env(f"{env_name}-v2", num_envs=1, wrapper=RewardInputNormalizeWrapper, wrapper_kwrags={'rwfn': rwfn})
        agent = FiniteSoftQiter(venv, gamma=1, alpha=0.01, device=rwfn.device)
        agent.learn(0)
        ax = fig.add_subplot(1, end_trial + 1, trial + 1)
        ag_im = ax.imshow(mapping(agent.policy.v_table[0].cpu().reshape(10, 10)), interpolation='None')
        fig.colorbar(ag_im)
    fig.tight_layout()
    plt.show()


def test_draw_costmap():
    inputs = [[[0, 1, 2, 3], [3, 4, 5, 6]]]
    fig = CostMap.draw_costmap(inputs)

from algos.torch.OptCont import LQRPolicy
class IDPLQRPolicy(LQRPolicy):
    def _build_env(self) -> np.array:
        I1, I2 = 0.878121, 1.047289
        l1 = 0.7970
        lc1, lc2 = 0.5084, 0.2814
        m1 ,m2 = 17.2955, 34.5085
        g = 9.81
        M = np.array([[I1 + m1*lc1**2 + I2 + m2*l1**2 + 2*m2*l1*lc2 + m2*lc2**2, I2 + m2*l1*lc2 + m2*lc2**2],
                      [I2 + m2*l1*lc2 + m2*lc2**2, I2 + m2*lc2**2]])
        C = np.array([[m1*lc1*g + m2*l1*g + m2*g*lc2, m2*g*lc2],
                      [m2*g*lc2, m2*g*lc2]])
        self.A, self.B = np.zeros([4, 4]), np.zeros([4, 2])
        self.A[:2, 2:] = np.eye(2, 2)
        self.A[2:, :2] = np.linalg.inv(M) @ C
        self.B[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
        self.Q = np.diag([3.5139, 0.2872182, 0.24639979, 0.01540204])
        self.R = np.diag([0.02537065/1600, 0.01358577/900])
        self.gear = 100

def test_reward_calculation():
    from imitation.data import types
    with open("../../IRL/demos/DiscretizedHuman/19191919_lqr/quadcost_from_contlqr_sub05.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    init_states = []
    for traj in expert_trajs:
        init_states += [traj.obs[0]]
    bsp = io.loadmat("../../IRL/demos/HPC/sub05/sub05i1.mat")['bsp']
    env = make_env("DiscretizedHuman-v0", bsp=bsp, N=[19, 19, 19, 19], NT=[11, 11], init_states=init_states)
    agent = FiniteSoftQiter(env, gamma=1, alpha=0.01, device='cpu')
    agent.learn(0)
    agent_trajs = []
    for i in range(len(init_states)):
        init_state = env.reset()
        obs, acts, rews = agent.predict(init_state, deterministic=True)
        data_dict = {'obs': obs, 'acts': acts, 'rews': rews.flatten(), 'infos': None}
        traj = types.TrajectoryWithRew(**data_dict)
        agent_trajs.append(traj)
        print(env.get_reward(expert_trajs[i].obs[:-1], expert_trajs[i].acts).sum(), rews.sum())
    print('end')


def test_cal_cost_for_airl():
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


def test_process_agent(tenv):
    cost_dir = os.path.join(irl_path, "tmp", "log", "HPC", "ppo", "AIRL_test", "69", "model")
    with open(cost_dir + "/discrim.pkl", "rb") as f:
        disc = pickle.load(f)
    cost_fn = disc.reward_net.base_reward_net.double()
    algo = PPO.load(cost_dir + "/gen.zip")
    agent = {"algo": algo, "env": tenv, "cost_fn": cost_fn}
    agents = CostMap.process_agent(agent)
    inputs = CostMap.cal_cost(agents)
    CostMap.draw_costmap(inputs)


def test_costmap(tenv):
    reward_dir = os.path.join(irl_path, "tmp", "log", "HPC", "ppo", "AIRL_test", "69", "model")
    with open(reward_dir + "/discrim.pkl", "rb") as f:
        disc = pickle.load(f).double()
    reward_fn = disc.reward_net.base_reward_net
    agent = PPO.load(reward_dir + "/gen.zip")
    # expt = def_policy("HPC", tenv)
    cost_map = CostMap(reward_fn, tenv, agent)

@pytest.mark.parametrize('trial', [1])
def test_learned_reward_mat(trial):
    from matplotlib import cm

    def feature_fn(x):
        # return x
        # return x ** 2
        return th.cat([x, x**2], dim=1)

    def normalized(x):
        return (x - x.min()) / (x.max() - x.min())

    env_name = "2DWorld_disc"
    expt_env = make_env(f"{env_name}-v2")
    log_path = f"{irl_path}/tmp/log/{env_name}/MaxEntIRL/ext_01alpha_nonorm_noact_lrdecay_10/01alpha_nobias_noact_many_{trial}"
    with open(f"{log_path}/model/reward_net.pkl", "rb") as f:
        rwfn = pickle.load(f)
    rwfn.feature_fn = feature_fn
    agent_env = make_env(f"{env_name}-v2", wrapper=RewardWrapper, wrapper_kwrags={'rwfn': rwfn.eval()})
    expt_reward_mat = normalized(expt_env.get_reward_mat())
    agent_reward_mat = normalized(agent_env.get_reward_mat().cpu().numpy())
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(np.vstack([expt_reward_mat[0, 10 * i:10 * (i + 1)] for i in range(10)]), cmap=cm.rainbow)
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(np.vstack([agent_reward_mat[0, 10 * i:10 * (i + 1)] for i in range(10)]), cmap=cm.rainbow)
    plt.show()
    print(np.abs(expt_reward_mat - agent_reward_mat).max())
