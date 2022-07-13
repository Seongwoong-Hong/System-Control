from functools import partial

import gym
import pytest

from algos.tabular.viter import backward_trans, forward_trans
from gym_envs.envs import DiscretizedPendulum

import numpy as np
import torch as th
from scipy.special import softmax, logsumexp


def test_discretized_pendulum():
    """ 기본 환경 테스트 """
    env = gym.make('DiscretizedPendulum-v0', N=[11, 11], NT=[11])  # type: DiscretizedPendulum

    # step test
    s = env.reset()
    # env.render()
    a = env.action_space.sample()
    next_s, r, _, _ = env.step(a)
    # print(f'Step from {s} to {next_s} by action {a}, torque {env.get_torque(a)}')

    # rendering test (P control)
    s = env.reset()
    for _ in range(100):
        a = np.array([0])
        s, r, d, _ = env.step(a)
        # env.render()

        if d:
            s = env.reset()


def test_calc_trans_mat():
    """
    근사된 전환 행렬 P 에 대한 value iteration 수행
    계산 시간, 초기 상태에 대한 평균 에러 계산
    """
    env = gym.make('DiscretePendulum-v2')  # type: DiscretizedPendulum

    h = [0.05, 0.05]
    env.get_trans_mat(h=h, verbose=True)


@pytest.mark.parametrize("soft", [True, False])
def test_value_itr(soft):
    """
    주어진 policy 에 대해 이산화된 전환 행렬 이용, value itr 수행
    greedy, soft update 구현됨
    """
    env = gym.make('DiscretizedPendulum-v2')  # type: DiscretizedPendulum
    soft = soft

    h = [0.1, 1]
    n_th, n_thd = env.get_num_cells(h)
    P = env.get_trans_mat(h)
    q_values = np.zeros([env.spec.max_episode_steps, env.num_actions, n_th * n_thd])

    def greedy_pi(s_array, q):
        # s_array.shape = (-1, 2), q.shape = (|A|, |S|), a_prob.shape = (|A|, -1)
        s_ind = env.get_ind_from_state(s_array, h=h, max_angle=env.max_angle, max_speed=env.max_speed)
        q_of_s = q @ s_ind.T
        a_prob = np.zeros_like(q_of_s)
        a_prob[np.argmax(q_of_s, axis=0).astype('i'), np.arange(q_of_s.shape[1])] = 1.0
        return a_prob

    def soft_pi(s_array, q):
        # s_array.shape = (-1, 2), q.shape = (|A|, |S|), a_prob.shape = (|A|, -1)
        s_ind = env.get_ind_from_state(s_array, h=h, max_angle=env.max_angle, max_speed=env.max_speed)
        q_of_s = q @ s_ind.T
        a_prob = softmax(q_of_s, axis=0)
        return a_prob

    # q learning iteration
    for itr in range(10):
        old_q = np.copy(q_values)

        for t_ind in reversed(range(env.spec.max_episode_steps)):
            if soft:
                pi = partial(soft_pi, q=q_values[t_ind])
            else:
                pi = partial(greedy_pi, q=q_values[t_ind])
            R = env.get_reward_vec()

            # q update
            if t_ind == env.spec.max_episode_steps - 1:
                q_values[t_ind] = R
            else:
                if soft:
                    next_values = logsumexp(q_values[t_ind + 1], axis=0)
                else:
                    # row-major 이기 때문에 action 선택  전치 수행
                    next_pi = partial(greedy_pi, q=q_values[t_ind + 1])
                    next_A = env.get_action_mat(next_pi, h)
                    next_values = q_values[t_ind + 1].T[next_A.T == 1]

                q_values[t_ind] = R + backward_trans(P, next_values)

        max_q_err = np.max(np.abs(old_q - q_values))
        assert not np.isnan(max_q_err)
        print(f'itr #{itr} Maximum Q err: {max_q_err:.2f}')

    # running with learned policy
    # todo: q-value 와 rollout value 의 갭 줄이는 방법
    import time
    s_vec, _ = env.get_vectorized(h)
    for itr in range(5):
        s = env.reset()
        print(f'Try #{itr} @ initial state {s}')
        appx_v, v = 0., 0.
        for t_ind in range(env.spec.max_episode_steps):
            s_ind = env.get_ind_from_state(s, h, env.max_angle, env.max_speed)
            a = np.argmax(q_values[t_ind] @ s_ind)
            next_s, r, _, _ = env.step(a)
            s = next_s
            v += r

            if soft:
                a_prob = pi(s[None, ...])
                v += - (a_prob * np.log(a_prob)).sum()

            if t_ind == 0:
                if soft:
                    appx_v = logsumexp(q_values[t_ind] @ s_ind, 0)
                else:
                    appx_v = np.max(q_values[t_ind] @ s_ind)

            env.render()
            time.sleep(0.01)

        print(f'{float(appx_v):.2f} (Q table) vs. {float(v):.2f} (rollout)')
    env.close()


def test_det():
    env = gym.make("DiscretizedPendulum-v0", h=[0.03, 0.15])
    while env.n <= len(env.init_state):
        env.reset()


def test_reward_for_wrapped_env():
    from common.wrappers import RewardWrapper
    from common.util import make_env
    from algos.torch.MaxEntIRL import RewardNet
    rwfn = RewardNet(inp=2, use_action_as_inp=False, feature_fn=lambda x: x ** 2, arch=[])
    env = make_env("DiscretizedPendulum-v2", h=[0.03, 0.15], num_envs=1, wrapper=RewardWrapper,
                   wrapper_kwrags={'rwfn': rwfn})
    env.env_method("get_reward_vec")
    assert True
