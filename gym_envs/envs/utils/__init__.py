import numpy as np


def calc_trans_mat_error(env, P, s_vec, a_vec, h, m, sampler):
    test_s = sampler(int(m))
    next_s = env.get_next_state(test_s, a_vec)
    test_s_ind = env.get_ind_from_state(test_s, h).T

    err = 0.
    for a_ind in range(len(a_vec)):
        next_s_pred = s_vec.T @ P[a_ind] @ test_s_ind
        err += np.mean(np.abs(next_s[a_ind] - next_s_pred.T))
    err /= len(a_vec)

    return err


def angle_normalize(x):
    """ 각 x 가 -pi ~ pi 사이 값으로 표현되도록 변환 """
    return ((x + np.pi) % (2 * np.pi)) - np.pi
