import numpy as np


def calc_trans_mat_error(env, P, s_vec, a_vec, h, t, sampler):
    test_s = sampler()
    test_s_pred = test_s
    err_list = []

    for _ in range(t):
        a_ind = np.random.choice(range(len(a_vec)))
        next_s = env.get_next_state(test_s, a_vec[[a_ind]])
        test_s_ind = env.get_ind_from_state(test_s_pred, h)
        next_s_pred = test_s_ind @ P[a_ind].T @ s_vec
        err = np.mean(np.abs(next_s - next_s_pred))
        err_list.append(err)
        test_s = next_s.squeeze()
        test_s_pred = next_s_pred.squeeze()

    return np.array(err_list)


def angle_normalize(x):
    """ 각 x 가 -pi ~ pi 사이 값으로 표현되도록 변환 """
    return ((x + np.pi) % (2 * np.pi)) - np.pi
