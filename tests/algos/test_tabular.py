import os.path
import pytest
import torch as th
import numpy as np
import pickle

from gym_envs.envs import DataBasedDiscretizationInfo
from common.util import make_env
from common.wrappers import RewardWrapper
from imitation.data.rollout import generate_trajectories, make_sample_until, flatten_trajectories
from imitation.util import logger
from algos.tabular.qlearning import *
from algos.tabular.viter import *
from matplotlib import pyplot as plt
from scipy import io

logger.configure(".", format_strs=['stdout'])


@pytest.fixture
def irl_path():
    return os.path.abspath("../../IRL")


@pytest.fixture
def bsp(irl_path):
    return io.loadmat(f"{irl_path}/demos/HPC/sub06/sub06i1.mat")['bsp']


def test_qlearning():
    env = make_env("2DTarget_disc-v2", map_size=10)
    logger.configure(".", format_strs=['stdout'])
    algo = QLearning(env, gamma=0.8, epsilon=0.4, alpha=0.4, device='cpu')
    algo.learn(int(1e4))

    print('end')


def test_soft_q_learning():
    env = make_env("2DTarget_disc-v2", map_size=10, num_envs=1)
    logger.configure(".", format_strs=['stdout'])
    algo = SoftQLearning(env, gamma=0.999, epsilon=0.4, alpha=0.01, device='cpu')
    algo.learn(int(1e4))

    print('end')


def test_viter():
    env = make_env("DiscretizedPendulum-v2", num_envs=1, N=[19, 19])
    algo = Viter(env, gamma=0.7, epsilon=0.2, device='cpu')
    algo.learn(2000)
    algo2 = FiniteViter(env, gamma=0.7, epsilon=0.2, device='cpu')
    algo2.learn(0)
    sample_until = make_sample_until(n_timesteps=None, n_episodes=10)
    infinite_trajs = generate_trajectories(algo, env, sample_until, deterministic_policy=True)

    obs_differs, acts_differs = [], []
    for traj in infinite_trajs:
        f_obs, f_acts, _ = algo2.predict(traj.obs[0], deterministic=True)
        obs_differs.append(np.abs(traj.obs - f_obs).mean())
        acts_differs.append(np.abs(traj.acts - f_acts).mean())

    assert np.array(obs_differs).mean() < 1e-2 and np.array(acts_differs).mean() < 0.1
    print(f"{np.array(obs_differs).mean()} < 1e-2 and {np.array(acts_differs).mean()} < 0.1")


def test_softqiter():
    env = make_env("2DTarget_disc-v2", map_size=10, num_envs=1)
    logger.configure(".", format_strs=['stdout'])
    algo = SoftQiter(env, gamma=0.999, alpha=0.01, device='cpu')
    algo.learn(int(1e4))
    algo2 = SoftQiter(env, gamma=0.999, alpha=0.05, device='cpu')
    algo2.learn(int(1e4))

    print('end')

def test_softiter_finite_diff():
    env = make_env("2DTarget_disc-v2", num_envs=1, map_size=10)
    logger.configure(".", format_strs=['stdout'])
    algo = SoftQiter(env, gamma=0.8, alpha=0.001, device='cuda:3')
    algo.learn(2000)
    algo2 = FiniteSoftQiter(env, gamma=0.8, alpha=0.001, device='cuda:3')
    algo2.learn(0)
    sample_until = make_sample_until(n_timesteps=None, n_episodes=10)
    infinite_trajs = generate_trajectories(algo, env, sample_until, deterministic_policy=True)

    obs_differs, acts_differs = [], []
    for traj in infinite_trajs:
        f_obs, f_acts, _ = algo2.predict(traj.obs[0], deterministic=True)
        obs_differs.append(np.abs(traj.obs - f_obs).mean())
        acts_differs.append(np.abs(traj.acts - f_acts).mean())

    assert np.array(obs_differs).mean() < 1e-2 and np.array(acts_differs).mean() < 0.1
    print(f"{np.array(obs_differs).mean()} < 1e-2 and {np.array(acts_differs).mean()} < 0.1")


def test_finite_iter(bsp):
    with open("../../IRL/demos/DiscretizedDoublePendulum/databased_lqr/obs_info_tree_5000.pkl", "rb") as f:
        obs_info_tree = pickle.load(f)
    with open("../../IRL/demos/DiscretizedDoublePendulum/databased_lqr/acts_info_tree_20.pkl", "rb") as f:
        acts_info_tree = pickle.load(f)
    obs_info = DiscretizationInfo([0.05, 0.05, 0.3, 0.35], [-0.05, -0.2, -0.08, -0.4], obs_info_tree)
    acts_info = DataBasedDiscretizationInfo([60, 50], [-60, -20], acts_info_tree)
    env = make_env("DiscretizedDoublePendulum-v2", obs_info=obs_info, acts_info=acts_info, num_envs=1)

    logger.configure(".", format_strs=['stdout'])
    algo = FiniteViter(env, gamma=1, alpha=0.00001, device='cpu')
    algo.learn(0)
    algo2 = FiniteSoftQiter(env, gamma=1, alpha=0.00001, device='cpu')
    algo2.learn(0)
    init_obs = env.reset()
    print(init_obs)
    obs1, acts1, rews1 = algo.predict(init_obs, deterministic=True)
    print(init_obs)
    obs2, acts2, rews2 = algo2.predict(init_obs, deterministic=True)
    assert np.abs(algo.policy.v_table - algo2.policy.v_table).mean() <= 1e-3
    assert np.abs(obs1 - obs2).mean() <= 1e-3
    assert np.abs(acts1 - acts2).mean() <= 1e-3
    assert np.abs(rews1 - rews2).mean() <= 1e-3


def test_infinite_iter(bsp):
    env = make_env("DiscretizedHuman-v2", N=[19, 17, 17, 17], NT=[11, 11], num_envs=1, bsp=bsp)
    algo = Viter(env, gamma=0.8, alpha=0.0001, device='cpu')
    algo.learn(300)
    algo2 = SoftQiter(env, gamma=0.8, alpha=0.001, device='cpu')
    algo2.learn(300)
    assert np.abs(algo.policy.v_table - algo2.policy.v_table).mean() <= 1e-3
    print(f"Value: {np.abs(algo.policy.v_table - algo2.policy.v_table).mean()}")


def test_state_visitation_diff_according_to_alpha_diff():
    from copy import deepcopy
    env = make_env("2DTarget_disc-v2", map_size=10)
    policy1 = FiniteSoftQiter(env, gamma=1, alpha=3, device='cpu')
    policy1.learn(0)
    policy2 = FiniteSoftQiter(env, gamma=1, alpha=6, device='cpu')
    policy2.learn(0)

    D_prev1 = th.ones([policy1.policy.obs_size]) / policy1.policy.obs_size
    D_prev2 = th.ones([policy2.policy.obs_size]) / policy2.policy.obs_size
    Dc1 = D_prev1[None, :] * policy1.policy.policy_table[0]
    Dc2 = D_prev2[None, :] * policy2.policy.policy_table[0]
    for t in range(1, policy1.max_t):
        D1 = th.zeros_like(D_prev1)
        D2 = th.zeros_like(D_prev2)
        for a in range(policy1.policy.act_size):
            D1 += policy1.transition_mat[a] @ (D_prev1 * policy1.policy.policy_table[t - 1, a])
            D2 += policy2.transition_mat[a] @ (D_prev2 * policy2.policy.policy_table[t - 1, a])
        Dc1 += policy1.policy.policy_table[t] * D1[None, :] * policy1.gamma ** t
        Dc2 += policy2.policy.policy_table[t] * D2[None, :] * policy2.gamma ** t
        D_prev1 = deepcopy(D1)
        D_prev2 = deepcopy(D2)

    fig1 = plt.figure(figsize=[8, 8], dpi=100)
    fig2 = plt.figure(figsize=[8, 8], dpi=100)
    for obs in range(100):
        ax1 = fig1.add_subplot(10, 10, obs+1)
        ax2 = fig2.add_subplot(10, 10, obs+1)
        ax1.imshow(Dc1[:, obs].reshape(7, 7))
        ax2.imshow(Dc2[:, obs].reshape(7, 7))
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()

    s_vec, a_vec = env.get_vectorized()

    def feature_fn(x):
        return x ** 2

    feat_mat = []
    for acts in a_vec:
        feat_mat.append(feature_fn(th.from_numpy(np.append(s_vec, np.repeat(acts[None, :], len(s_vec), axis=0), axis=1))).numpy())
    mean_features1 = np.sum(np.sum(Dc1.numpy()[..., None] * np.array(feat_mat), axis=0), axis=0)
    mean_features2 = np.sum(np.sum(Dc2.numpy()[..., None] * np.array(feat_mat), axis=0), axis=0)
    print(mean_features1)
    print(mean_features2)
    print((Dc1 - Dc2).abs().mean())


def test_wrapped_reward():
    def feature_fn(x):
        return th.cat([x, x**2], dim=1)

    reward_dir = "../../IRL/tmp/log/1DTarget_disc/MaxEntIRL/ext_ppo_disc_samp_linear_ppoagent_svm_reset/model"
    with open(f"{reward_dir}/000/reward_net.pkl", "rb") as f:
        reward_net = pickle.load(f).cpu()
    reward_net.feature_fn = feature_fn
    env = make_env("1DTarget_disc-v2", wrapper=RewardWrapper, wrapper_kwrags={'rwfn': reward_net})
    algo1 = QLearning(env, gamma=0.8, epsilon=0.2, alpha=0.6, device='cpu')
    algo1.learn(1000)
    algo2 = Viter(env, gamma=0.8, epsilon=0.2, device='cpu')
    algo2.learn(50)

    print('env')

