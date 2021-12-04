import time

import gym
import numpy as np
import matplotlib.pyplot as plt
import gym_envs
from algos.tabular.viter import forward_trans, backward_trans
from gym_envs.envs import TwoDTargetCont


def test_2d_target_cont():
    """ 기본 환경 테스트 """
    env = gym.make('2DTarget_cont-v1')          # type: TwoDTargetCont

    s = env.reset()
    a = env.get_optimal_action(s)
    print(f'For target {env.target}, optimal action in {s} is {a}!')


def test_optimal_value():
    """
    optimal policy 의 value 계산, value contour 그림 생성
    target=(2., 2.) 에서 value 가 가장 높은 것 확인, optimal policy insanity check
    """
    env = gym.make('2DTarget_cont-v1')
    s = env.reset()

    init_states = np.zeros([300, 2])
    values = np.zeros(300)
    for itr in range(300):                      # sampling initial state
        init_states[itr] = s
        v_epi = 0.
        while True:                             # rollout a episode
            a = env.get_optimal_action(s)
            s, r, d, _ = env.step(a)
            v_epi += r

            if d:
                s = env.reset()
                break
        values[itr] = v_epi
    plt.tricontour(init_states[:, 0], init_states[:, 1], values)
    plt.show()
    print(f'\n'
          f'Mean value: {np.mean(values):.2f} \n'
          f'Max value {np.max(values):.2f} @ {init_states[np.argmax(values)]} \n'
          f'Min value {np.min(values):.2f} @ {init_states[np.argmin(values)]}')


def test_cal_trans_mat():
    """
    근사된 전환 행렬 P 에 대한 value iteration 수행
    계산 시간, 초기 상태에 대한 평균 에러 계산
    """
    env = gym.make('2DTarget_cont-v1')          # type: TwoDTargetCont

    for h in [0.1, 0.05, 0.01]:
        t_start = time.time()
        _ = env.get_trans_mat(h=h, verbose=True)
        print(f'(h={h:.2f}) execution takes {time.time() - t_start:.2f} sec')


def test_value_itr():
    """
    주어진 policy 에 대해 전환 행렬 P_pi 계산, value itr 수행
    optimal policy 와의 value 차이 계산
    NOTE:: state diff (+- 1) 가 h 배수이면 value 차이는 매우 작음
    """
    env = gym.make('2DTarget_cont-v1')          # type: TwoDTargetCont

    opt_pi = env.get_optimal_policy()
    h = 0.007
    N = round(env.map_size / h) + 1
    P = env.get_trans_mat(h=h)
    A = env.get_action_mat(opt_pi, h=h)
    R = env.get_reward_vec(opt_pi, h=h)
    values = np.zeros([env.spec.max_episode_steps, N ** 2])

    # do value iteration with optimal policy and approximated dynamics
    for t_ind in range(99, -1, -1):
        if t_ind == 99:
            values[t_ind] = R
        else:
            values[t_ind] = R + backward_trans(P, A, values[t_ind + 1])

    # get true value by roll-out with optimal
    s_vec = np.stack(np.meshgrid(h * np.arange(0., N),
                                 h * np.arange(0., N),
                                 indexing='xy'),
                     -1).reshape(-1, 2)
    s_vec_origin = np.copy(s_vec)

    true_values = np.zeros(N ** 2)
    for _ in range(env.spec.max_episode_steps):
        a_vec = env.get_optimal_action_array(s_vec)
        true_values += env.get_reward(s_vec, a_vec)
        s_vec = env.get_next_state(s_vec, a_vec)

    # calc and plot value error
    err = np.abs(true_values - values[0])
    plt.figure(figsize=(3, 3))
    plt.tricontour(s_vec_origin[:, 0], s_vec_origin[:, 1], err)
    plt.title(f'Value gap by dyn appx (h={h:.3f})')
    plt.colorbar()
    plt.show()

    print(f'\n'
          f':: Appx dyn vs. True VI :: \n'
          f'Err max: {err.max():.2E} @ {s_vec_origin[np.argmax(err)]}')
