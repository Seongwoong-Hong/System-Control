import pytest
import torch as th
import pickle
from common.util import make_env
from common.wrappers import RewardWrapper
from algos.tabular.qlearning import QLearning
from algos.tabular.viter import Viter


def test_qlearning():
    env = make_env("1DTarget_disc-v2")
    algo = QLearning(env, gamma=0.8, epsilon=0.2, alpha=0.9, device='cpu')
    algo.learn(1000)

    print('end')


def test_viter():
    env = make_env("1DTarget_disc-v2")
    algo = Viter(env, gamma=0.8, epsilon=0.2, device='cpu')
    algo.learn(200)

    print('env')


def test_wrapped_reward():
    with open("../../IRL/tmp/log/1DTarget_disc/MaxEntIRL/ext_ppo_disc_samp_linear_ppoagent_svm_reset/model/000/reward_net.pkl", "rb") as f:
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