import numpy as np


def forward_trans(P, A, v):
    """
    정책 A 와 전환 행렬 P 를 고려할 때 다음 스텝의 v 계산
    P.shape = (|A|, |S|, |S|)
    A.shape = (|A|, |S|)
    next_v.shape = (|S|)
    """
    num_actions = P.shape[0]

    next_v = 0.
    for a_ind in range(num_actions):
        next_v += P[a_ind] @ (A[a_ind] * v)[..., None]

    return next_v.ravel()


def backward_trans(P, v):
    """
    모든 A 에 대해 전환 행렬 P 를 고려할 때 이전 스텝의 v 계산
    P.shape = (|A|, |S|, |S|)
    A.shape = (|A|, |S|)
    post_v.shape = (|A|, |S|)
    """
    num_actions = P.shape[0]

    post_v = []
    for a_ind in range(num_actions):
        post_v.append((P[a_ind].T @ v))

    post_v = np.stack(post_v, axis=0)

    return post_v


from algos.tabular.viter.viter import Viter, SoftQiter, FiniteViter, FiniteSoftQiter
