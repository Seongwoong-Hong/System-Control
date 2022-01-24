import os.path

import pytest
import torch as th
import numpy as np
import pickle
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
    env = make_env("2DTarget_disc-v2", map_size=7, num_envs=10)
    logger.configure(".", format_strs=['stdout'])
    algo = SoftQLearning(env, gamma=0.8, epsilon=0.4, alpha=0.2, device='cpu')
    import time
    start = time.time()
    algo.learn(int(1e4))
    print(f"time: {time.time() - start}")

    print('end')


def test_viter():
    env = make_env("2DTarget_disc-v2", map_size=10, num_envs=1)
    logger.configure(".", format_strs=['stdout'])
    algo = Viter(env, gamma=0.8, epsilon=0.2, device='cpu')
    algo.learn(2000)

    print('env')


def test_softiter():
    env = make_env("DiscretizedDoublePendulum-v2", num_envs=1, N=[19, 17, 17, 17])
    logger.configure(".", format_strs=['stdout'])
    algo = SoftQiter(env, gamma=0.8, alpha=0.001, device='cuda:2')
    algo.learn(2000)
    algo2 = FiniteSoftQiter(env, gamma=0.8, alpha=0.001, device='cuda:2')
    algo2.learn(0)
    sample_until = make_sample_until(n_timesteps=None, n_episodes=10)
    infinite_trajs = generate_trajectories(algo, env, sample_until, deterministic_policy=True)

    obs_differs, acts_differs = [], []
    for traj in infinite_trajs:
        f_obs, f_acts, _ = algo2.predict(traj.obs[0], deterministic=True)
        obs_differs.append(np.abs(traj.obs[:-1, :] - f_obs))
        acts_differs.append(np.abs(traj.acts - f_acts))

    assert np.array(obs_differs).mean() < 1e-2 and np.array(acts_differs).mean() < 0.1


def test_finite_iter(bsp):
    env = make_env("DiscretizedHuman-v2", N=[19, 17, 17, 17], NT=[11, 11], num_envs=1, bsp=bsp)
    logger.configure(".", format_strs=['stdout'])
    algo = FiniteViter(env, gamma=0.8, alpha=0.00001, device='cpu')
    algo.learn(0)
    algo2 = FiniteSoftQiter(env, gamma=0.8, alpha=0.00001, device='cpu')
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


def test_wrapped_reward():
    with open(
            "../../IRL/tmp/log/1DTarget_disc/MaxEntIRL/ext_ppo_disc_samp_linear_ppoagent_svm_reset/model/000/reward_net.pkl",
            "rb") as f:
        reward_net = pickle.load(f).cpu()
    env = make_env("1DTarget_disc-v2", wrapper=RewardWrapper, wrapper_kwrags={'rwfn': reward_net})
    algo1 = QLearning(env, gamma=0.8, epsilon=0.2, alpha=0.6, device='cpu')
    algo1.learn(1000)
    algo2 = Viter(env, gamma=0.8, epsilon=0.2, device='cpu')
    algo2.learn(50)

    print('env')


if __name__ == "__main__":
    def feature_fn(x):
        return th.cat([x, x**2], dim=1)

    test_wrapped_reward()