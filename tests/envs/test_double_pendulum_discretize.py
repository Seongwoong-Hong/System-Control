from functools import partial

import gym
from scipy.special import softmax, logsumexp

from algos.tabular.viter import backward_trans, forward_trans
from gym_envs.envs.double_pendulum_discretized import DiscretizedDoublePendulum

import numpy as np


def test_discretized_pendulum():
    """ 기본 환경 테스트 """
    env = gym.make('DiscretizedDoublePendulum-v0', N=[21, 21, 11, 11])  # type: DiscretizedDoublePendulum

    # step test
    s = env.reset()
    a = env.action_space.sample()

    env.render()

    next_s, r, _, _ = env.step(a)
    print(f'Step from {s} to {next_s} by action {a}, torque {env.get_torque(a)}')

    # rendering test (P control)
    s = env.reset()
    for _ in range(400):
        a = np.array([0, 0])
        s, r, d, _ = env.step(a)
        env.render()

        if d:
            s = env.reset()
    env.close()


def test_calc_trans_mat():
    """
    근사된 전환 행렬 P 에 대한 value iteration 수행
    계산 시간, 초기 상태에 대한 평균 에러 계산
    """
    N = [21, 21, 11, 11]
    env = gym.make('DiscretizedDoublePendulum-v2', N=N)  # type: DiscretizedDoublePendulum

    # h = [0.1, 0.1, 0.1, 0.1]
    # h = [0.1, 0.1, 1.0, 1.0]
    env.get_trans_mat(h=None, verbose=True)


# @pytest.mark.parametrize("soft", [True, False])
def test_value_itr(soft=True):
    """
    주어진 policy 에 대해 이산화된 전환 행렬 이용, value itr 수행
    greedy, soft update 구현됨
    """
    import time, os
    from scipy import io

    N = [21, 21, 11, 11]
    env = gym.make('DiscretizedDoublePendulum-v2', N=N)  # type: DiscretizedDoublePendulum

    n_dim = np.prod(env.get_num_cells())
    P = env.get_trans_mat()
    q_values = np.zeros([env.spec.max_episode_steps, np.prod(env.num_actions), n_dim])

    def greedy_pi(s_array, q):
        # s_array.shape = (-1, 2), q.shape = (|A|, |S|), a_prob.shape = (|A|, -1)
        s_ind = env.get_ind_from_state(s_array)
        q_of_s = q @ s_ind.T
        a_prob = np.zeros_like(q_of_s)
        a_prob[np.argmax(q_of_s, axis=0).astype('i'), np.arange(q_of_s.shape[1])] = 1.0
        return a_prob

    def soft_pi(s_array, q):
        # s_array.shape = (-1, 2), q.shape = (|A|, |S|), a_prob.shape = (|A|, -1)
        s_ind = env.get_ind_from_state(s_array)
        q_of_s = q @ s_ind.T
        a_prob = softmax(q_of_s, axis=0)
        return a_prob

    # q learning iteration
    for itr in range(1):
        t1 = time.time()
        old_q = np.copy(q_values)

        for t_ind in reversed(range(env.spec.max_episode_steps)):
            if soft:
                pi = partial(soft_pi, q=q_values[t_ind])
            else:
                pi = partial(greedy_pi, q=q_values[t_ind])
            R = env.get_reward_mat()

            # q update
            if t_ind == env.spec.max_episode_steps - 1:
                q_values[t_ind] = R
            else:
                if soft:
                    next_values = logsumexp(q_values[t_ind + 1], axis=0)
                else:
                    # row-major 이기 때문에 action 선택  전치 수행
                    next_pi = partial(greedy_pi, q=q_values[t_ind + 1])
                    next_A = env.get_action_mat(next_pi)
                    next_values = q_values[t_ind + 1].T[next_A.T == 1]

                q_values[t_ind] = R + backward_trans(P, v=next_values)
        print(time.time() - t1)
        max_q_err = np.max(np.abs(old_q - q_values))
        print(f'itr #{itr} Maximum Q err: {max_q_err:.2f}')

    # running with learned policy
    # todo: q-value 와 rollout value 의 갭 줄이는 방법
    s_vec, a_vec = env.get_vectorized()

    for itr in range(5):
        s = env.reset()
        print(f'Try #{itr} @ initial state {s}')
        appx_v, v = 0., 0.
        for t_ind in range(env.spec.max_episode_steps):
            s_ind = env.get_ind_from_state(s)
            a = np.argmax(q_values[t_ind] @ s_ind)
            next_s, r, _, _ = env.step(a_vec[a])
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
            time.sleep(0.05)
            env.render()

        print(f'{float(appx_v):.2f} (Q table) vs. {float(v):.2f} (rollout)')
    env.close()
