from algos.tabular.viter.viter import Viter, SoftQiter, FiniteSoftQiter


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

def backward_trans(P, A, v):
    """
    정책 A 와 전환 행렬 P 를 고려할 때 이전 스텝의 v 계산
    P.shape = (|A|, |S|, |S|)
    A.shape = (|A|, |S|)
    post_v.shape = (|S|)
    """
    num_actions = P.shape[0]

    post_v = 0.
    for a_ind in range(num_actions):
        post_v += A[a_ind] * (P[a_ind].T @ v)

    return post_v
