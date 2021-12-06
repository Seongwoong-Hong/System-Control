from os import path
from sys import getsizeof

import gym
import numpy as np
from scipy.sparse import csc_matrix


class TwoDTargetCont(gym.Env):
    """
    연속적 상태의 결정적 2 차원 점 환경
    고정 타겟과의 거리에 반비례하여 보상 계산
    타겟으로 최단 거리로 움직이는 optimal action, policy 생성 method 포함
    이산화되는 경우를 위해 전환 행렬 계산 method 포함
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, map_size=10):
        super(TwoDTargetCont, self).__init__()
        self.map_size = map_size
        self.state = np.array([0., 0.])
        self.target = np.array([2., 2.])
        self.observation_space = gym.spaces.Box(low=np.array([0., 0.]), high=np.array([map_size, map_size]))
        self.action_space = gym.spaces.MultiDiscrete([3, 3])

        self.viewer = None

    def reset(self):
        self.state = self.observation_space.sample()
        return self.get_obs()

    def step(self, action: np.ndarray):
        assert self.state is not None, "Can't step the environment before calling reset function"
        assert action in self.action_space, f"{action} is Out of action space"
        r = self.get_reward(self.state, action)
        self.state = self.get_next_state(self.state, action)

        return self.get_obs(), r, None, {}

    def set_state(self, state):
        self.state = state

    def get_reward(self, state, action):
        return - np.linalg.norm(state - self.target, axis=-1) ** 2

    def get_next_state(self, state, action):
        diff = action - 1
        next_s = np.clip(state + diff, a_min=0., a_max=self.map_size)
        return next_s

    def get_obs(self):
        return self.state

    def get_optimal_action(self, state):
        """ 타겟까지 최단거리로 움직이는 행동 생성 """
        to_target = np.round(self.target - state)
        opt_action = [np.sign(to_target[0]), np.sign(to_target[1])]
        return np.array(opt_action) + 1

    def get_optimal_action_array(self, s_array):
        """ state array 를 입력받았을 때 optimal actions 반환 """
        to_target = np.round(self.target - s_array)
        opt_actions = np.stack([np.sign(to_target[:, 0]), np.sign(to_target[:, 1])], -1)
        return opt_actions + 1

    def get_optimal_policy(self):
        """
        상태를 입력받아 9 가지 action 의 확률을 반환하는 정책 리턴
        xy-indexing 을 따라 vector 로 변환되며, action space 에 속하지 않음
        policy: (x, y) -> (p0, p1, ..., p8)
        """
        def optimal_policy(s_array):
            """
            s.shape = (-1, 2)
            a_prob.shape = (-1, 9)
            """
            a_prob = np.zeros([s_array.shape[0], 9])
            opt_a_array = self.get_optimal_action_array(s_array)
            opt_a_ind = 3 * opt_a_array[:, 1] + opt_a_array[:, 0]
            a_prob[np.arange(s_array.shape[0]), opt_a_ind.astype('i')] = 1.0
            return a_prob

        return optimal_policy

    def get_num_cells(self, h):
        """
        분해능 h 로 이산화된 state 의 dimension 별 수 반환
        즉, |S| = n_x * n_y
        """
        h_x, h_y = h
        n_x = round(self.map_size / h_x) + 1
        n_y = round(self.map_size / h_y) + 1

        return n_x, n_y

    def get_vectorized(self, h):
        """
        분해능 h 로 이산화된 state 과 이산 action 의 vectorization 수행 및 반환
        vectorization 순서는 meshgrid xy indexing 을 따름
        s_vec.shape = (|S|, 2)
        a_vec.shape = (9, 2)
        """
        h_x, h_y = h
        n_x, n_y = self.get_num_cells(h)

        s_vec = np.stack(np.meshgrid(h_x * np.arange(0., n_x), h_y * np.arange(0., n_y), indexing='xy'),
                         -1).reshape(-1, 2)
        a_vec = np.stack(np.meshgrid([0, 1, 2], [0, 1, 2], indexing='xy'),
                         -1).reshape(-1, 2)

        return s_vec, a_vec

    def get_ind_from_state(self, state, h):
        """
        분해능 h 로 vectorized 되었을 때, 입력 state 의 index 반환
        state.shape = (-1, 2)
        ind_vec.shape = (-1, |S|)
        """
        h_th, h_thd = h
        n_th, n_thd = self.get_num_cells(h)
        x_vec, y_vec = np.split(state, 2, axis=-1)

        x_idx = np.round(x_vec / h_th).astype('i')
        y_idx = np.round(y_vec / h_thd).astype('i')
        tot_idx = n_th * y_idx + x_idx

        if state.ndim == 1:
            ind_vec = np.zeros(n_th * n_thd).astype('i')
            ind_vec[tot_idx.ravel()] = 1
        else:
            batch_size = state.shape[0]
            ind_vec = csc_matrix((np.ones(batch_size), (np.arange(batch_size), tot_idx.ravel())),
                                 shape=[batch_size, n_th * n_thd])

        return ind_vec

    def get_trans_mat(self, h, verbose=False):
        """
        분해능 h 주어질 때 모든 action 대한 전환 행렬 계산 (zero-hold 방식)
        P[i, j, k] 은 i-th action 을 했을 때, j-th -> k-th state 로 옮겨갈 확률임
        verbose=True 일 때 uniform 초기 상태에 대한 평균 에러 계산

        P.shape = (|A|, |S|, |S|)
        """
        s_vec, a_vec = self.get_vectorized(h)
        next_s_vec_list = [self.get_next_state(s_vec, a) for a in a_vec]

        P = np.stack([self.get_ind_from_state(next_s_vec, h).T for next_s_vec in next_s_vec_list], 0)

        if verbose:
            # calculate mean error by discretization
            M = 10 ** 3
            test_s = np.random.uniform(low=0., high=self.map_size, size=[M, 2])
            next_s = np.stack([self.get_next_state(test_s, a) for a in a_vec], 0)
            test_s_ind = self.get_ind_from_state(test_s, h).T

            err = 0.
            for a_ind in range(9):
                next_s_pred = s_vec.T @ P[a_ind] @ test_s_ind
                err += np.mean(np.abs(next_s[a_ind] - next_s_pred.T))
            err /= 9
            print(f'(h={h}) 1 step prediction error: {err:.4f}')

        return P

    def get_action_mat(self, pi, h):
        """
        정책 pi, 분해능 h 가 주어졌을 때 forward 을 위한 action matrix, A 계산
        A[i, j] 는 j-th state 에서 i-th action 을 할 확률을 뜻함
        고차원에서 적은 메모리 사용을 위해 필요하며 forward 는 아래같이 계산됨 (forward_trans 참고)
        next_v = sum(P @ A .* v, axis=0) by forward_trans(P, A, v) in algo/tabular/viter/__init__.py

        pi: (-1, 2) -> (|A|, -1)
        A.shape = (|A|, |S|)
        """
        s_vec, _ = self.get_vectorized(h)
        return pi(s_vec)

    def get_reward_vec(self, pi, h, soft=False):
        """
        정책 pi, 분해능 h 에 대해 모든 상태의 보상 계산, 벡터로 반환
        R.shape = (|S|)
        """
        s_vec, _ = self.get_vectorized(h)
        R = self.get_reward(s_vec, None)

        if soft:
            a_prob = pi(s_vec)
            R += - (a_prob * np.log(a_prob)).sum(axis=0)

        return R

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-.2, self.map_size + .2, -.2, self.map_size + .2)

            for x in range(self.map_size + 1):
                vertical = rendering.Line(start=(x, 0.), end=(x, self.map_size))
                horizontal = rendering.Line(start=(0., x), end=(self.map_size, x))
                self.viewer.add_geom(vertical)
                self.viewer.add_geom(horizontal)

            goal = rendering.make_circle(0.5)
            goal.set_color(0.3, 0.8, 0.3)
            goal_transform = rendering.Transform()
            goal.add_attr(goal_transform)
            goal_transform.set_translation(self.target[0], self.target[1])
            self.viewer.add_geom(goal)

            agent = rendering.make_circle(0.5)
            agent.set_color(0.3, 0.3, 0.8)
            self.agent_transform = rendering.Transform()
            agent.add_attr(self.agent_transform)
            self.viewer.add_geom(agent)

        self.agent_transform.set_translation(self.state[0], self.state[1])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

