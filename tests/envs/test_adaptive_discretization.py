import pickle
import pytest
import time
import numpy as np
import torch as th

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from gym_envs.envs import DataBasedDiscretizationInfo, UncorrDiscretizationInfo, FaissDiscretizationInfo
from gym_envs.envs.base.info_tree import *
from algos.torch.OptCont import LQRPolicy
from algos.tabular.viter import FiniteSoftQiter
from common.util import make_env
from common.wrappers import DiscretizeWrapper
from demos.MakeInfoTree import IPLQRPolicy, DiscIDPLQRPolicy


def test_information_tree():
    info_tree = InformationTree(np.array([1, 2]), 1)
    print(info_tree.data, info_tree.data[0].value, info_tree.data[0].radius)


def test_dividing_nodes():
    print("\n")
    info_tree = InformationTree(np.array([1, 2]), np.array([1, 1]))
    for _ in range(10):
        target = np.random.uniform(low=np.array([0, 1]), high=np.array([2, 3]))
        info_tree.divide_node(target)
    print(len(info_tree.data))
    for node in info_tree.data:
        print(node.value, node.radius, node.visitation)


def test_draw_discrete_node():
    with open("../../demos/DiscretizedPendulum/databased_lqr/obs_info_tree_200.pkl", "rb") as f:
        info_tree = pickle.load(f)
    ax = plt.gca()
    for node in info_tree.data:
        rect = Rectangle(node.value - node.radius, node.radius[0] * 2, node.radius[1] * 2, linewidth=0.5, edgecolor='k', facecolor='none')
        # rect = Rectangle((node.value - node.radius, 0), node.radius * 2, 2, linewidth=0.5, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
    plt.xlim([-0.05, 0.05])
    # plt.xlim([-30, 40])
    plt.ylim([-0.08, 0.3])
    # plt.ylim([0, 2])
    plt.show()


@pytest.mark.parametrize("trial", [10, 50, 100, 200, 500, 1000])
def test_data_based_discretization(trial):
    obs_high = np.array([0.05, 0.05, 0.3, 0.35])
    obs_low = np.array([-0.05, -0.2, -0.08, -0.4])
    acts_high = np.array([60., 50])
    acts_low = np.array([-60., -20])
    init_state = (obs_high + obs_low) / 2
    radius = (obs_high - obs_low) / 2
    obs_info_tree = InformationTree(init_state, radius, FourDStateNode)
    init_act = (acts_high + acts_low) / 2
    act_rad = (acts_high - acts_low) / 2
    acts_info_tree = InformationTree(init_act, act_rad, TwoDStateNode)
    obs_info = DataBasedDiscretizationInfo(obs_high, obs_low, obs_info_tree)
    acts_info = DataBasedDiscretizationInfo(acts_high, acts_low, acts_info_tree)
    env = make_env("DiscretizedDoublePendulum-v2", obs_info=obs_info, acts_info=acts_info, wrapper=DiscretizeWrapper)
    agent = DiscIDPLQRPolicy(env=env)
    start = time.time()
    for ep in range(trial):
        agent.set_env(env)
        for _ in range(50):
            ob = env.reset()
            done = False
            while not done:
                obs_info_tree.count_visitation(ob)
                act, _ = agent.predict(ob)
                acts_info_tree.count_visitation(act)
                ob, _, done, _ = env.step(act)
        for _ in range(1):
            max_idx = np.argmax(obs_info_tree.visitation)
            obs_info_tree.divide_node(obs_info_tree.data[max_idx])
        if ep % 7 == 0:
            max_idx = np.argmax(acts_info_tree.visitation)
            acts_info_tree.divide_node(acts_info_tree.data[max_idx])
        env.obs_info.set_info(env.obs_high, env.obs_low, obs_info_tree)
        env.acts_info.set_info(env.max_torques, env.min_torques, acts_info_tree)
    print(time.time() - start)


def test_divide_node():
    init_state = [0.1, 0.2, 0.4, 0.8]
    radius = [0.05, 0.1, 0.05, 0.02]
    info_tree = InformationTree(init_state, radius, FourDStateNode)
    for ep in range(2):
        max_idx = np.argmax(info_tree.visitation)
        info_tree.divide_node(info_tree.data[max_idx])
    print(info_tree.data)


def test_import_info_tree():
    with open("../../demos/DiscretizedPendulum/databased_lqr/obs_info_tree_1500.pkl", "rb") as f:
        info_tree = pickle.load(f)
    obs_info = DataBasedDiscretizationInfo([0.05, 0.3], [-0.05, -0.08], info_tree)
    obs = obs_info.get_vectorized()
    print(len(obs))


def test_compare_index():
    with open("../../demos/DiscretizedDoublePendulum/databased_lqr/obs_info_tree_5000.pkl", "rb") as f:
        obs_info_tree = pickle.load(f)
    obs_high = np.array([0.05, 0.05, 0.3, 0.35])
    obs_low = np.array([-0.05, -0.2, -0.08, -0.4])
    obs_info1 = FaissDiscretizationInfo(obs_high, obs_low, obs_info_tree)
    obs_info2 = DataBasedDiscretizationInfo(obs_high, obs_low, obs_info_tree)
    for _ in range(100):
        info = np.random.uniform(low=obs_low, high=obs_high).reshape(1, -1)
        assert obs_info1.get_idx_from_info(info) == obs_info2.get_idx_from_info(info)


def test_learning_time():
    with open("../../demos/DiscretizedDoublePendulum/databased_lqr/obs_info_tree_15000.pkl", "rb") as f:
        obs_info_tree = pickle.load(f)
    with open("../../demos/DiscretizedDoublePendulum/databased_lqr/acts_info_tree_60.pkl", "rb") as f:
        acts_info_tree = pickle.load(f)
    s = time.time()
    obs_high = np.array([0.05, 0.05, 0.3, 0.35])
    obs_low = np.array([-0.05, -0.2, -0.08, -0.4])
    acts_high = np.array([60., 50.])
    acts_low = np.array([-60., -20., ])
    obs_info = FaissDiscretizationInfo(obs_high, obs_low, obs_info_tree)
    acts_info = FaissDiscretizationInfo(acts_high, acts_low, acts_info_tree)
    env = make_env("DiscretizedDoublePendulum-v2", obs_info=obs_info, acts_info=acts_info, wrapper=DiscretizeWrapper)
    agent = FiniteSoftQiter(env, gamma=1, alpha=0.0001, device='cpu')
    agent.learn(0)
    print(time.time() - s)
