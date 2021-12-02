import pytest
import torch as th
import pickle
from common.util import make_env
from common.wrappers import RewardWrapper
from imitation.util import logger
from algos.tabular.qlearning import *
from algos.tabular.viter import *

logger.configure(".", format_strs=['stdout'])


def test_qlearning():
    env = make_env("2DTarget_disc-v2", map_size=10)
    logger.configure(".", format_strs=['stdout'])
    algo = QLearning(env, gamma=0.8, epsilon=0.4, alpha=0.4, device='cpu')
    algo.learn(int(1e4))

    print('end')


def test_soft_q_learning():
    env = make_env("2DTarget_disc-v2", map_size=7, use_vec_env=True, num_envs=10)
    logger.configure(".", format_strs=['stdout'])
    algo = SoftQLearning(env, gamma=0.8, epsilon=0.4, alpha=0.2, device='cpu')
    import time
    start = time.time()
    algo.learn(int(1e4))
    print(f"time: {time.time() - start}")

    print('end')


def test_viter():
    env = make_env("2DTarget_disc-v2", map_size=10, use_vec_env=True, num_envs=1)
    logger.configure(".", format_strs=['stdout'])
    algo = Viter(env, gamma=0.8, epsilon=0.2, device='cpu')
    algo.learn(2000)

    print('env')


def test_softiter():
    env = make_env("2DTarget_disc-v2", map_size=10, use_vec_env=True, num_envs=1)
    logger.configure(".", format_strs=['stdout'])
    algo = SoftQiter(env, gamma=0.8, epsilon=0.2, alpha=1, device='cpu')
    algo.learn(2000)

    print('env')


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