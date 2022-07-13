import pickle
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from gym_envs.envs import InformationTree, TwoDStateNode, OneDStateNode
from algos.torch.OptCont import LQRPolicy
from common.util import make_env
from common.wrappers import DiscretizeWrapper


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
    with open("acts_test.pkl", "rb") as f:
        info_tree = pickle.load(f)
    ax = plt.gca()
    for node in info_tree.data:
        rect = Rectangle((node.value - node.radius, 0), node.radius[0] * 2, 2, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
    plt.xlim([-30, 40])
    plt.ylim([0, 2])
    plt.show()



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
        self.Q = np.diag([2.8139, 1.04872182])
        self.R = np.diag([1.617065e-4])
        self.gear = 1


def test_data_based_discretization():
    env = make_env("DiscretizedPendulum_DataBased-v2", N=[29, 29], NT=[51], wrapper=DiscretizeWrapper)
    agent = IPLQRPolicy(env=make_env("DiscretizedPendulum_DataBased-v2", N=[29, 29], NT=[51]))
    init_state = (env.obs_high + env.obs_low) / 2
    radius = (env.obs_high - env.obs_low) / 2
    obs_info_tree = InformationTree(init_state, radius, TwoDStateNode)
    init_act = (env.max_torques + env.min_torques) / 2
    act_rad = (env.max_torques - env.min_torques) / 2
    acts_info_tree = InformationTree(init_act, act_rad, OneDStateNode)
    for ep in range(50):
        env.obs_info.set_info(env.obs_high, env.obs_low, obs_info_tree)
        env.acts_info.set_info(env.max_torques, env.min_torques, acts_info_tree)
        agent.set_env(env)
        for _ in range(5):
            ob = env.reset()
            done = False
            while not done:
                obs_info_tree.find_target_node(ob)
                act, _ = agent.predict(ob)
                acts_info_tree.find_target_node(act)
                ob, _, done, _ = env.step(act)
        for _ in range(5):
            max_idx = np.argmax(obs_info_tree.visitation)
            obs_info_tree.divide_node(obs_info_tree.data[max_idx].value)
        if ep % 2 == 0:
            max_idx = np.argmax(acts_info_tree.visitation)
            acts_info_tree.divide_node(acts_info_tree.data[max_idx].value)
    with open("obs_test.pkl", "wb") as f:
        pickle.dump(obs_info_tree, f)
    with open("acts_test.pkl", "wb") as f:
        pickle.dump(acts_info_tree, f)


def test_divide_node():
    env = make_env("DiscretizedPendulum_DataBased-v2", N=[29, 29], NT=[21], wrapper=DiscretizeWrapper)
    init_state = (env.obs_high + env.obs_low) / 2
    radius = (env.obs_high - env.obs_low) / 2
    info_tree = InformationTree(init_state, radius)
    for ep in range(200):
        max_idx = np.argmax(info_tree.visitation)
        info_tree.divide_node(info_tree.data[max_idx].value)
    print(info_tree.data)


def test_import_info_tree():
    with open("test.pkl", "rb") as f:
        info_tree = pickle.load(f)
    env = make_env("DiscretizedPendulum_DataBased-v2", N=[29, 29], NT=[21], wrapper=DiscretizeWrapper)
    env.obs_info.info_tree = info_tree
    obs, acts = env.get_vectorized()
    print(obs)