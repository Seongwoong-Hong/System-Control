import os
import pytest
import pickle
import json
import time
import numpy as np
import torch as th
from scipy import io, signal
from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from algos.tabular.viter import FiniteSoftQiter, FiniteViter, SoftQiter
from algos.torch.OptCont import LQRPolicy, FiniteLQRPolicy, DiscreteLQRPolicy
from gym_envs.envs import FaissDiscretizationInfo, UncorrDiscretizationInfo
from common.util import make_env, CPU_Unpickler
from common.verification import verify_policy
from common.wrappers import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from stable_baselines3.common.vec_env import DummyVecEnv


@pytest.fixture
def rl_path():
    return os.path.abspath(os.path.join("..", "..", "RL"))


def test_mujoco_envs_learned_policy():
    env_name = "Pendulum"
    env = make_env(f"{env_name}-v0", use_vec_env=False)
    name = f"{env_name}/ppo"
    model_dir = os.path.join("..", "..", "RL", "mujoco_envs", "tmp", "log", name)
    algo = SAC.load(model_dir + "/agent")
    a_list, o_list, _ = verify_policy(env, algo, render='human', repeat_num=10)


class IPLQRPolicy(LQRPolicy):
    def _build_env(self):
        g = 9.81
        m = 17.2955
        l = 0.7970
        lc = 0.5084
        I = 0.878121 + m * lc**2
        self.A, self.B = np.zeros([2, 2]), np.zeros([2, 1])
        self.A[0, 1] = 1
        self.A[1, 0] = m * g * lc / I
        self.B[1, 0] = 1 / I
        self.Q = self.env.envs[0].Q
        self.R = self.env.envs[0].R
        self.gear = 1

    # def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
    #     action = super(DiscIPLQRPolicy, self)._predict(observation[:, :2], deterministic)
    #     high = th.from_numpy(self.env.action_space.high).float()
    #     low = th.from_numpy(self.env.action_space.low).float()
    #     action = th.clip(action, min=low, max=high)
    #     d_act = self.env.env_method("get_acts_from_idx", self.env.env_method("get_idx_from_acts", action.numpy())[0])[0]
    #     return th.from_numpy(d_act).float()


class IDPLQRPolicy(LQRPolicy):
    def _build_env(self) -> np.array:
        I1, I2 = self.env.envs[0].Is
        l1 = self.env.envs[0].ls[0]
        lc1, lc2 = self.env.envs[0].lcs
        m1 ,m2 = self.env.envs[0].ms
        g = 9.81
        M = np.array([[I1 + m1*lc1**2 + I2 + m2*l1**2 + 2*m2*l1*lc2 + m2*lc2**2, I2 + m2*l1*lc2 + m2*lc2**2],
                      [I2 + m2*l1*lc2 + m2*lc2**2, I2 + m2*lc2**2]])
        C = np.array([[m1*lc1*g + m2*l1*g + m2*g*lc2, m2*g*lc2],
                      [m2*g*lc2, m2*g*lc2]])
        self.A, self.B = np.zeros([4, 4]), np.zeros([4, 2])
        self.A[:2, 2:] = np.eye(2, 2)
        self.A[2:, :2] = np.linalg.inv(M) @ C
        self.B[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
        # self.A, self.B, _, _, dt = signal.cont2discrete((self.A, self.B, np.array([1, 1, 1, 1]), 0), self.env.envs[0].dt)
        self.Q = self.env.envs[0].Q
        self.R = self.env.envs[0].R
        # self.gear = th.tensor([60, 50])
        self.gear = 1

    # def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
    #     action = super(DiscIDPLQRPolicy, self)._predict(observation, deterministic)
    #     high = th.from_numpy(self.env.action_space.high).float()
    #     low = th.from_numpy(self.env.action_space.low).float()
    #     action = th.clip(action, min=low, max=high)
    #     d_act = self.env.env_method("get_acts_from_idx", self.env.env_method("get_idx_from_acts", action.numpy())[0])[0]
    #     return th.from_numpy(d_act).float()


def test_rl_learned_policy(rl_path):
    env_type = "IDP"
    name = f"{env_type}_custom"
    subj = "sub05"
    irl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "IRL"))
    subpath = os.path.join(irl_dir, "demos", "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    with open(f"{irl_dir}/demos/DiscretizedHuman/19191919/{subj}_1.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    init_states = []
    for traj in expt_trajs:
        init_states += [traj.obs[0]]
    with open(f"../../IRL/demos/DiscretizedHuman/databased_lqr/obs_info_tree_4000.pkl", "rb") as f:
        obs_info_tree = pickle.load(f)
    with open(f"../../IRL/demos/DiscretizedHuman/databased_lqr/acts_info_tree_80.pkl", "rb") as f:
        acts_info_tree = pickle.load(f)
    # obs_info = FaissDiscretizationInfo([0.02, 0.05, 0.3, 0.45], [-0.01, -0.2, -0.18, -0.4], obs_info_tree)
    # acts_info = FaissDiscretizationInfo([20, 20], [-40, -10], acts_info_tree)
    obs_info = UncorrDiscretizationInfo([0.05, 0.05, 0.3, 0.45], [-0.05, -0.2, -0.18, -0.4], [19, 19, 19, 19])
    acts_info = UncorrDiscretizationInfo([100, 100], [-100, -100], [11, 11])
    env = make_env(f"DiscretizedHuman-v2", obs_info=obs_info, acts_info=acts_info, bsp=bsp)#, init_states=init_states)
    # env = make_env(f"DiscretizedPendulum-v2", obs_info=obs_info, acts_info=acts_info, wrapper=DiscretizeWrapper)
    env.Q = np.diag([0.7139, 0.5872182, 1.0639979, 0.9540204])
    env.R = np.diag([.0061537065, .0031358577])
    # env.Q = np.diag([0.7139, 1.0639979])
    # env.R = np.diag([.0061537065])
    algo = IDPLQRPolicy(env)
    # algo = SoftQiter(env, gamma=0.9995, alpha=0.0001,device='cuda:0')
    # algo.learn(5e4)
    # env = make_env(f"{name}-v2", bsp=bsp)
    name += f"_{subj}"
    model_dir = os.path.join(rl_path, env_type, "tmp", "log", name, "ppo", "policies_17")
    stats_path = None
    if os.path.isfile(model_dir + "normalization.pkl"):
        stats_path = model_dir + "normalization.pkl"
    # with open(model_dir + "/agent.pkl", "rb") as f:
    #     algo = pickle.load(f)
    # algo.set_env(env)
    # algo = PPO.load(model_dir + f"/agent_7")
    fig = plt.figure(figsize=[9.6, 9.6])
    ax11 = fig.add_subplot(3, 2, 1)
    ax12 = fig.add_subplot(3, 2, 2)
    ax21 = fig.add_subplot(3, 2, 3)
    ax22 = fig.add_subplot(3, 2, 4)
    ax31 = fig.add_subplot(3, 2, 5)
    ax32 = fig.add_subplot(3, 2, 6)
    rews = []
    for _ in range(300):
        obs, acts = [], []
        ob = env.reset()
        obs.append(ob)
        done = False
        rew = 0
        for _ in range(100):
            a, _ = algo.predict(ob, deterministic=True)
            # a = env.get_acts_from_idx(env.get_idx_from_acts(np.array([a * np.array([60, 50])])))[0]
            ob, rw, done, _ = env.step(a)
            obs.append(ob)
            acts.append(a)
            # env.render()
            # time.sleep(env.dt)
            rew += rw
        rews.append(rew)
        obs = np.array(obs)
        acts = np.array(acts)
        ax11.plot(obs[:-1, 0])
        ax12.plot(obs[:-1, 1])
        ax11.set_ylim([-.05, .05])
        ax12.set_ylim([-.2, .05])
        ax21.plot(obs[:-1, 2])
        ax22.plot(obs[:-1, 3])
        ax21.set_ylim([-.18, .3])
        ax22.set_ylim([-.4, .45])
        ax31.plot(acts[:, 0])
        ax32.plot(acts[:, 1])
        # ax31.set_ylim([-30, 40])
        ax31.set_ylim([-60, 60])
        ax32.set_ylim([-20, 50])
    fig.tight_layout()
    plt.show()
    print(np.mean(rews), np.std(rews))
    env.close()


class FiniteIDPLQRPolicy(FiniteLQRPolicy):
    def _build_env(self) -> np.array:
        I1, I2 = self.env.envs[0].Is
        l1 = self.env.envs[0].ls[0]
        lc1, lc2 = self.env.envs[0].lcs
        m1, m2 = self.env.envs[0].ms
        g = 9.81
        M = np.array([[I1 + m1*lc1**2 + I2 + m2*l1**2 + 2*m2*l1*lc2 + m2*lc2**2, I2 + m2*l1*lc2 + m2*lc2**2],
                      [I2 + m2*l1*lc2 + m2*lc2**2, I2 + m2*lc2**2]])
        C = np.array([[m1*lc1*g + m2*l1*g + m2*g*lc2, m2*g*lc2],
                      [m2*g*lc2, m2*g*lc2]])
        self.A, self.B = np.zeros([4, 4]), np.zeros([4, 2])
        self.A[:2, 2:] = np.eye(2, 2)
        self.A[2:, :2] = np.linalg.inv(M) @ C
        self.B[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
        self.A, self.B, _, _, dt = signal.cont2discrete((self.A, self.B, np.array([1, 1, 1, 1]), 0), self.env.envs[0].dt)
        print(dt)
        # self.A = self.A * self.env.envs[0].dt + np.eye(4, 4)
        # self.B = self.B * self.env.envs[0].dt
        self.Q = self.env.envs[0].Q*100
        self.R = self.env.envs[0].R*100
        # self.gear = th.tensor([60, 50])
        self.gear = 1


def test_finite_algo(rl_path):
    env_type = "IDP"
    env_id = f"{env_type}_custom"
    subj = "sub05"
    irl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "IRL"))
    subpath = os.path.join(irl_dir, "demos", "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    # with open(f"../../IRL/demos/{env_type}/databased_faiss_400080/{subj}_1.pkl", "rb") as f:
    #     expt_trajs = pickle.load(f)
    # init_states = []
    # for traj in expt_trajs:
    #     init_states.append(traj.obs[0])
    # with open(f"../../IRL/demos/{env_type}/databased_lqr/obs_info_tree_4000.pkl", "rb") as f:
    #     obs_info_tree = pickle.load(f)
    # with open(f"../../IRL/demos/{env_type}/databased_lqr/acts_info_tree_80.pkl", "rb") as f:
    #     acts_info_tree = pickle.load(f)
    obs_info = UncorrDiscretizationInfo([0.05, 0.05, 0.3, 0.45], [-0.05, -0.2, -0.18, -0.4], [19, 19, 19, 19])
    acts_info = UncorrDiscretizationInfo([100, 100], [-100, -100], [11, 11])
    env = make_env(f"DiscretizedHuman-v2", obs_info=obs_info, acts_info=acts_info, bsp=bsp)#, init_states=init_states)
    env.Q = np.diag([1.9139, 1.5872182, 2.0639979, 1.9540204])
    env.R = np.diag([.00061537065, .00031358577])
    agent = FiniteIDPLQRPolicy(env)
    env = make_env(f"{env_id}-v2", bsp=bsp)
    agent.set_env(env)
    # agent = FiniteSoftQiter(env, gamma=0.9995, alpha=0.0001, device='cpu')
    # agent.learn(0)
    fig = plt.figure(figsize=[9.6, 9.6])
    ax11 = fig.add_subplot(3, 2, 1)
    ax12 = fig.add_subplot(3, 2, 2)
    ax21 = fig.add_subplot(3, 2, 3)
    ax22 = fig.add_subplot(3, 2, 4)
    ax31 = fig.add_subplot(3, 2, 5)
    ax32 = fig.add_subplot(3, 2, 6)
    rews = []
    b, a = signal.butter(3, 3.5/100)
    for epi in range(300):
        # ob = init_states[epi % len(init_states)]
        ob = env.reset()
        obs, acts, rws = agent.predict(ob, deterministic=False)
        # obs = signal.filtfilt(b, a, obs, axis=0)
        # acts = signal.filtfilt(b, a, acts, axis=0)
        rews.append(rws.sum())
        ax11.plot(obs[:-1, 0])
        ax12.plot(obs[:-1, 1])
        ax11.set_ylim([-.05, .05])
        ax12.set_ylim([-.2, .05])
        ax21.plot(obs[:-1, 2])
        ax22.plot(obs[:-1, 3])
        ax21.set_ylim([-.18, .3])
        ax22.set_ylim([-.4, .45])
        ax31.plot(acts[:, 0])
        ax32.plot(acts[:, 1])
        # ax31.set_ylim([-30, 40])
        ax31.set_ylim([-60, 60])
        ax32.set_ylim([-20, 50])
        # env.render()
        # tot_r = 0
        # for t in range(40):
        #     obs_idx = env.get_idx_from_obs(ob)
        #     act_idx = agent.policy.choice_act(agent.policy.policy_table[t].T[obs_idx])
        #     act = env.get_acts_from_idx(act_idx)
        #     ob, r, _, _ = env.step(act[0])
        #     tot_r += r
        #     env.render()
        #     time.sleep(env.dt)
        # print(tot_r)
    print(np.mean(rews), np.std(rews))
    fig.tight_layout()
    plt.show()
    print('end')


def test_toy_disc_env(rl_path):
    name = "SpringBall"
    env_id = f"{name}_disc"
    env = make_env(f"{env_id}-v2", wrapper=DiscretizeWrapper)
    algo = SoftQiter(env=env, gamma=0.99, alpha=0.01, device='cpu')
    algo.learn(1000)
    fig = plt.figure(figsize=[6.4, 6.4])
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    for _ in range(10):
        obs = []
        acts = []
        ob = env.reset()
        done = False
        # env.render()
        while not done:
            obs.append(ob)
            act, _ = algo.predict(ob, deterministic=False)
            ob, _, done, _ = env.step(act[0])
            acts.append(act[0])
            # env.render()
            # time.sleep(env.dt)
        obs = np.array(obs)
        acts = np.array(acts)
        ax1.plot(obs[:, 0])
        ax2.plot(obs[:, 1])
        ax3.plot(acts)
    fig.tight_layout()
    plt.show()


def test_1d(rl_path):
    name = "1DTarget_disc"
    env_id = f"{name}"
    map_size = 50
    env = make_env(f"{env_id}-v2", map_size=map_size, num_envs=1)
    model_dir = os.path.join(rl_path, name, "tmp", "log", env_id, "softqiter", "policies_1")
    with open(model_dir + "/agent.pkl", "rb") as f:
        algo = pickle.load(f)
    # algo = PPO.load(model_dir + "/agent")
    plt.imshow(algo.policy.policy_table, cmap=cm.rainbow)
    plt.show()
    print('end')


def test_total_reward(rl_path):
    env_type = "HPC"
    name = f"{env_type}_custom"
    model_dir = os.path.join(rl_path, env_type, "tmp", "log", name, "ppo", "policies_4")
    stats_path = None
    if os.path.isfile(model_dir + "normalization.pkl"):
        stats_path = model_dir + "normalization.pkl"
    env = make_env(f"{name}-v1", subpath="../../IRL/demos/HPC/sub01/sub01", wrapper=ActionWrapper, use_norm=stats_path)
    algo = PPO.load(model_dir + f"/agent")
    ob = env.reset()
    done = False
    actions = []
    obs = []
    rewards = []
    while not done:
        act, _ = algo.predict(ob, deterministic=False)
        ob, reward, done, info = env.step(act)
        rewards.append(reward)
        actions.append(info['acts'])
        obs.append(info['obs'])
    # plt.plot(np.array(actions).reshape(-1, 2))
    # plt.show()
    # plt.plot(np.array(obs).reshape(-1, 6)[:, :2])
    # plt.show()
    plt.plot(rewards)
    plt.show()
