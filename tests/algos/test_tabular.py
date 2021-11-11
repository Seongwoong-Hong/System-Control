import pytest
from common.util import make_env
from algos.tabular.qlearning import QLearning
from algos.tabular.viter import Viter


def test_qlearning():
    env = make_env("1DTarget_disc-v2")
    algo = QLearning(env, 0.8, 0.2, 0.9, device='cpu')
    algo.learn(10000)

    print('end')


def test_viter():
    env = make_env("1DTarget_disc-v2")
    algo = Viter(env, 0.8, 0.2)
    algo.learn(10)

    print('env')
