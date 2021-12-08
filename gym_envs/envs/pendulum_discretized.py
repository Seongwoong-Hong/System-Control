import gym
from copy import deepcopy
import numpy as np
from gym.utils import seeding
from scipy.sparse import csc_matrix
from os import path


class DiscretizedPendulum(gym.Env):
    """
    State, action 이 이산화된 Single Pendulum 환경, 전환 행렬 P 계산
    Reward 는 state 에 대한 2 차 cost 의 음수임
    Step 은 explicit euler 방법을 따름
    Angle 과 speed 는 maximum 값에서 clip 됨
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, h=None):
        super(DiscretizedPendulum, self).__init__()
        self.max_torque = 5.0  # gym version 보다 큼, swing 을 사용하지 않기 때문
        self.max_speed = 1.0
        self.max_angle = np.pi / 3
        self.dt = 0.05
        self.g = 9.81
        self.m = 1.0
        self.l = 1.0
        self.h = h
        self.num_actions = 7

        self.np_random = None
        self.state = None
        self.viewer = None
        self.last_a = None

        obs_high = np.array([self.max_angle, self.max_speed]) + self.h
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([7])
        self.torque_list = np.linspace(-self.max_torque, self.max_torque, self.num_actions)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        high = np.array([self.max_angle, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_a = None

        return self.get_obs()

    def step(self, action: np.ndarray):
        assert self.state is not None, "Can't step the environment before calling reset function"
        assert action in self.action_space, f"{action} is Out of action space"
        self.last_a = action
        info = {'obs': self.state.reshape(1, -1), 'act': action.reshape(1, -1)}
        r = self.get_reward(self.state, action)
        self.state = self.get_next_state(self.state, action)

        return self.get_obs(), r, False, info

    def set_state(self, state: np.ndarray):
        assert state in self.observation_space
        self.state = state

    def get_torque(self, action):
        return self.torque_list[action.item()]

    def get_reward(self, state, action):
        if state.ndim == 1:
            th, thd = state
        else:
            th, thd = np.split(state, 2, axis=-1)
        return - angle_normalize(th) ** 2 - 0.1 * thd ** 2

    def get_next_state(self, state, action):
        """
        Explicit euler 방식 으로 ode integration 수행
        """
        th, thd = np.split(state, 2, axis=-1)
        torque = self.get_torque(action)
        torque = torque[..., None] if state.ndim == 2 else torque
        g, m, l, dt = self.g, self.m, self.l, self.dt

        new_thd = thd + (3 / 2 * g / l * np.sin(th) + 3 / (m * l ** 2) * torque) * dt
        new_th = th + new_thd * dt

        new_th = np.clip(new_th, -self.max_angle, self.max_angle)
        new_thd = np.clip(new_thd, -self.max_speed, self.max_speed)

        if state.ndim == 1:
            return np.array([new_th[0], new_thd[0]])
        else:
            return np.column_stack([new_th, new_thd])

    def get_obs(self):
        return self.state

    def get_num_cells(self, h=None):
        """
        분해능 h 로 이산화된 state 의 dimension 별 수 반환
        즉, |S| = n_x * n_y
        """
        if h is None:
            h = self.h
        assert h is not None, "Env doesn't have resolution, h should be specified"
        h_th, h_thd = h
        n_th = round(2 * self.max_angle / h_th) + 1
        n_thd = round(2 * self.max_speed / h_thd) + 1

        return n_th, n_thd

    def get_vectorized(self, h=None):
        """
        분해능 h 로 이산화된 state 과 이산 action 의 vectorization 수행 및 반환
        vectorization 순서는 meshgrid xy indexing 을 따름
        s_vec.shape = (|S|, 2)
        a_vec.shape = (|A|, 2)
        """
        if h is None:
            h = self.h
        assert h is not None, "Env doesn't have resolution, h should be specified"
        h_th, h_thd = h
        n_th, n_thd = self.get_num_cells(h)

        s_vec = np.stack(np.meshgrid(h_th * np.arange(0., n_th) - self.max_angle,
                                     h_thd * np.arange(0., n_thd) - self.max_speed,
                                     indexing='xy'
                                     ),
                         -1).reshape(-1, 2)
        a_vec = np.arange(self.num_actions)

        return s_vec, a_vec

    # TODO: max_angle과 max_speed가 외부에서 입력될 필요가 있나?
    def get_ind_from_state(self, state, h=None, max_angle=None, max_speed=None):
        """
        분해능 h 로 vectorized 되었을 때, 입력 state 의 index 반환
        state.shape = (-1, 2)
        ind_vec.shape = (-1, |S|)
        """
        if h is None:
            h = self.h
        assert h is not None, "Env doesn't have resolution, h should be specified"
        h_th, h_thd = h
        n_th, n_thd = self.get_num_cells(h)
        th_vec, thd_vec = np.split(state, 2, axis=-1)

        th_idx = np.round((th_vec + self.max_angle) / h_th).astype('i')
        thd_idx = np.round((thd_vec + self.max_speed) / h_thd).astype('i')
        tot_idx = n_th * thd_idx + th_idx

        if state.ndim == 1:
            ind_vec = np.zeros(n_th * n_thd).astype('i')
            ind_vec[tot_idx.ravel()] = 1
        else:
            batch_size = state.shape[0]
            ind_vec = csc_matrix((np.ones(batch_size), (np.arange(batch_size), tot_idx.ravel())),
                                 shape=[batch_size, n_th * n_thd])

        return ind_vec

    def get_idx_from_obs(self, obs: np.ndarray):
        h_th, h_thd = self.h
        n_th, n_thd = self.get_num_cells(self.h)
        th_vec, thd_vec = np.split(obs, 2, axis=-1)

        th_idx = np.round((th_vec + self.max_angle) / h_th).astype('i')
        thd_idx = np.round((thd_vec + self.max_speed) / h_thd).astype('i')
        tot_idx = n_th * thd_idx + th_idx
        return tot_idx.flatten()

    def get_act_from_idx(self, idx: np.ndarray):
        return idx.reshape(-1, 1)

    def get_trans_mat(self, h=None, verbose=False):
        """
        분해능 h 주어질 때 모든 action 대한 전환 행렬 계산 (zero-hold 방식)
        P[i, j, k] 은 i-th action 을 했을 때, k-th -> j-th state 로 옮겨갈 확률임
        verbose=True 일 때 uniform 초기 상태에 대한 평균 에러 계산

        P.shape = (|A|, |S|, |S|)
        """
        if h is None:
            h = self.h
        assert h is not None, "Env doesn't have resolution, h should be specified"
        s_vec, a_vec = self.get_vectorized(h)
        next_s_vec_list = [self.get_next_state(s_vec, a) for a in a_vec]
        P = np.stack([self.get_ind_from_state(next_s_vec, h, self.max_angle, self.max_speed).T
                      for next_s_vec in next_s_vec_list], 0)

        if verbose:
            M = 10 ** 3
            high = np.array([self.max_angle, 1])
            test_s = np.random.uniform(low=-high, high=high, size=[M, 2])
            next_s = np.stack([self.get_next_state(test_s, a) for a in a_vec], 0)
            test_s_ind = self.get_ind_from_state(test_s, h, self.max_angle, self.max_speed).T

            err = 0.
            for a_ind in range(self.num_actions):
                next_s_pred = s_vec.T @ P[a_ind] @ test_s_ind
                err += np.mean(np.linalg.norm(next_s[a_ind] - next_s_pred.T, axis=-1))
            err /= self.num_actions
            print(f'(h={h}) 1 step prediction error: {err:.4f}')

        return P

    def get_action_mat(self, pi, h=None):
        """
        정책 pi, 분해능 h 가 주어졌을 때 forward 을 위한 action matrix, A 계산
        A[i, j] 는 j-th state 에서 i-th action 을 할 확률을 뜻함
        고차원에서 적은 메모리 사용을 위해 필요하며 forward 는 아래같이 계산됨 (forward_trans 참고)
        next_v = sum(P @ A .* v, axis=0) by forward_trans(P, A, v) in algo/tabular/viter/__init__.py

        pi: (-1, 2) -> (|A|, -1)
        A.shape = (|A|, |S|)
        """
        if h is None:
            h = self.h
        assert h is not None, "Env doesn't have resolution, h should be specified"
        s_vec, _ = self.get_vectorized(h)
        return pi(s_vec)

    def get_reward_vec(self, pi=None, h=None, soft=False):
        """
        정책 pi, 분해능 h 에 대해 모든 상태의 보상 계산, 벡터로 반환
        R.shape = (|S|)
        """
        if h is None:
            h = self.h
        assert h is not None, "Env doesn't have resolution, h should be specified"
        s_vec, a_vec = self.get_vectorized(h)
        R = self.get_reward(s_vec, a_vec).ravel()

        if soft:
            assert pi is not None, "Soft version needs policy pi"
            a_prob = pi(s_vec)
            R += - (a_prob * np.log(a_prob)).sum(axis=0)

        return R

    def render(self, mode="human"):
        # adapted from gym/envs/classic_control/pendulum.py
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(gym.__file__), 'envs',
                              'classic_control', 'assets', 'clockwise.png')
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_a:
            torque = self.get_torque(self.last_a)
            self.imgtrans.scale = (- torque / 2, np.abs(torque) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        # adapted from gym/envs/classic_control/pendulum.py
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class DiscretizedPendulumDet(DiscretizedPendulum):
    def __init__(self, h=None):
        super(DiscretizedPendulumDet, self).__init__(h=h)
        self.init_state, _ = super(DiscretizedPendulumDet, self).get_vectorized()
        self.init_state = self.init_state[0:len(self.init_state):5]
        self.n = 0

    def reset(self):
        self.set_state(self.init_state[self.n % len(self.init_state)])
        self.n += 1
        self.last_a = None
        return self.get_obs()

    def get_vectorized(self, h=None):
        s_vec = deepcopy(self.init_state)
        a_vec = np.arange(self.num_actions)
        return s_vec, a_vec


def angle_normalize(x):
    """ 각 x 가 -pi ~ pi 사이 값으로 표현되도록 변환 """
    return ((x + np.pi) % (2 * np.pi)) - np.pi
