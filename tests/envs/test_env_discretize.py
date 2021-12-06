import time
from functools import partial

import gym
import numpy as np
import pytest
from scipy.special import softmax, logsumexp

from algos.tabular.viter import backward_trans
from gym_envs.envs import TwoDTargetCont


def test_2d_target_cont():
    """ 기본 환경 테스트 """
    env = gym.make('2DTarget_cont-v1')          # type: TwoDTargetCont

    s = env.reset()
    a = env.get_optimal_action(s)
    print(f'For target {env.target}, optimal action in {s} is {a}!')

    # rendering test (P control)
    s = env.reset()
    for _ in range(100):
        a = env.get_optimal_action(s)
        s, r, d, _ = env.step(a)
        env.render()

        if d:
            s = env.reset()


def test_cal_trans_mat():
    """
    근사된 전환 행렬 P 에 대한 value iteration 수행
    계산 시간, 초기 상태에 대한 평균 에러 계산
    """
    env = gym.make('2DTarget_cont-v1')          # type: TwoDTargetCont

    h = [0.05, 0.05]
    t_start = time.time()
    _ = env.get_trans_mat(h=h, verbose=True)
    print(f'(h={h}) execution takes {time.time() - t_start:.2f} sec')


@pytest.mark.parametrize("soft", [True, False])
def test_value_itr(soft):
    """
    주어진 policy 에 대해 이산화된 전환 행렬 이용, value itr 수행
    greedy, soft update 구현됨
    """
    env = gym.make('2DTarget_cont-v1')          # type: TwoDTargetCont
    soft = soft

    h = [0.05, 0.05]
    n_x, n_y = env.get_num_cells(h)
    P = env.get_trans_mat(h=h)
    q_values = np.zeros([env.spec.max_episode_steps, 9, n_x * n_y])

    def greedy_pi(s_array, q):
        # s_array.shape = (-1, 2), q.shape = (|A|, |S|), a_prob.shape = (|A|, -1)
        s_ind = env.get_ind_from_state(s_array, h=h)
        q_of_s = q @ s_ind.T
        a_prob = np.zeros_like(q_of_s)
        a_prob[np.argmax(q_of_s, axis=0).astype('i'), np.arange(q_of_s.shape[1])] = 1.0
        return a_prob

    def soft_pi(s_array, q):
        # s_array.shape = (-1, 2), q.shape = (|A|, |S|), a_prob.shape = (|A|, -1)
        s_ind = env.get_ind_from_state(s_array, h=h)
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
            R = env.get_reward_vec(pi, h, soft=soft)

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
        print(f'itr #{itr} Maximum Q err: {max_q_err:.2f}')

    # running with learned policy
    s_vec, _ = env.get_vectorized(h)
    for itr in range(5):
        s = env.reset()
        print(f'Try #{itr} @ initial state {s}')
        appx_v, v = 0., 0.
        for t_ind in range(env.spec.max_episode_steps):
            s_ind = env.get_ind_from_state(s, h)
            a_ind = np.argmax(q_values[t_ind] @ s_ind)
            next_s, r, _, _ = env.step(np.array([a_ind % 3, a_ind // 3]))
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

        print(f'{float(appx_v):.2f} (Q table) vs. {float(v):.2f} (rollout)')
    env.close()
