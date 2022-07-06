import gym
import numpy as np
from os import path
from copy import deepcopy
from scipy.sparse import csc_matrix
from gym_envs.envs import BaseDiscEnv


class DiscretizedPendulum(BaseDiscEnv):
    """
    State, action 이 이산화된 Single Pendulum 환경, 전환 행렬 P 계산
    Reward 는 state 에 대한 2 차 cost 의 음수임
    Step 은 explicit euler 방법을 따름
    Angle 과 speed 는 bound_info.json 값에서 clip 됨
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, N, NT):
        super(DiscretizedPendulum, self).__init__()
        self.dt = 0.025
        self.g = 9.81
        self.m = 5.
        self.l = 1.
        self.lc = self.l / 2
        self.I = self.m * self.l ** 2 / 3
        self.num_actions = np.array(NT)
        self.Q = np.diag([3.5139, 1.2872182])
        self.R = np.diag([0.02537065])

        self.max_angle = None
        self.max_speed = None
        self.min_angle = None
        self.min_speed = None

        self.state = None
        self.viewer = None
        self.last_a = None

        if N is int:
            assert N % 2, "N should be a odd number"
            N = N * np.ones(4, dtype=int)
        else:
            N = np.array(N)
            assert N is not None, "The number of discretization should be defined"
            assert (N % 2).all(), "N should be consist of odd numbers"
        self.num_cells = N
        self.set_bounds(
            max_states=[0.10, 0.3],
            min_states=[-0.10, -0.3],
            max_torques=[50.],
            min_torques=[-30.],
        )

    def reset(self):
        high = np.array([*self.max_angle, *self.max_speed])
        low = np.array([*self.min_angle, *self.min_speed])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs()

    def step(self, action: np.ndarray):
        assert self.state is not None, "Can't step the environment before calling reset function"
        assert action in self.action_space, f"{action} is Out of action space"
        self.last_a = action
        info = {'obs': self.state.reshape(1, -1), 'act': action.reshape(1, -1)}
        r = self.get_reward(self.state, action)
        self.state = self.get_next_state(self.state, action)

        return self._get_obs(), r, False, info

    def set_state(self, state: np.ndarray):
        assert state in self.observation_space
        self.state = state

    def set_bounds(self, max_states, min_states, max_torques, min_torques):
        self.max_angle = np.array(max_states)[:1]
        self.max_speed = np.array(max_states)[1:]
        self.min_angle = np.array(min_states)[:1]
        self.min_speed = np.array(min_states)[1:]
        self.max_torques = np.array(max_torques)
        self.min_torques = np.array(min_torques)

        self.obs_high = np.array([*self.max_angle, *self.max_speed])
        self.obs_low = np.array([*self.min_angle, *self.min_speed])

        self.obs_list = []
        for high, low, n in zip(self.obs_high, self.obs_low, self.num_cells):
            self.obs_list.append(np.linspace(low, high, n + 1))
        self.acts_list = []
        for high, low, n in zip(self.max_torques, self.min_torques, self.num_actions):
            self.acts_list.append(np.linspace(low, high, n + 1))
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float64)
        self.action_space = gym.spaces.Box(low=self.min_torques, high=self.max_torques, dtype=np.float64)

    def get_reward(self, state, action):
        state = deepcopy(state)
        norm_torques = action / self.max_torques[None, ...]
        if state.ndim == 1:
            state[0] = angle_normalize(state[0])
            state[1] = angle_normalize(state[1])
            r = - np.sum((state[None, ...] @ self.Q) * state, axis=-1)
        else:
            state[:, 0] = angle_normalize(state[:, 0])
            state[:, 1] = angle_normalize(state[:, 1])
            r = - np.sum((state @ self.Q) * state, axis=-1)
        r -= np.sum((norm_torques @ self.R) * norm_torques, axis=-1)
        return r

    def get_next_state(self, state, action):
        """
        Explicit euler 방식 으로 ode integration 수행
        """
        th, thd = np.split(state, 2, axis=-1)
        torque = action.T
        torque = torque[..., None] if state.ndim == 2 else torque

        new_thd = thd + ((self.m * self.g * self.lc) / self.I * np.sin(th) +  torque / self.I) * self.dt
        new_th = th + new_thd * self.dt

        new_th = np.clip(new_th, self.min_angle, self.max_angle)
        new_thd = np.clip(new_thd, self.min_speed, self.max_speed)

        if state.ndim == 1:
            return np.array([new_th[0], new_thd[0]])
        else:
            return np.column_stack([new_th, new_thd])

    def _get_obs(self):
        return self.state

    def get_init_vector(self):
        return self.get_vectorized()

    def get_ind_from_state(self, state):
        state_backup = deepcopy(state)
        if len(state.shape) == 1:
            state = state[None, :]
        dims = self.num_cells
        tot_idx = self.get_idx_from_obs(state)

        if state_backup.ndim == 1:
            ind_vec = np.zeros(np.prod(dims)).astype('i')
            ind_vec[tot_idx.ravel()] = 1
        else:
            batch_size = state.shape[0]
            ind_vec = csc_matrix((np.ones(batch_size), (np.arange(batch_size), tot_idx.ravel())),
                                 shape=[batch_size, np.prod(dims)])

        return ind_vec

    def get_trans_mat(self, h=None, verbose=False):
        # Transition Matrix shape : (|A|, |Next S|, |S|)
        s_vec, a_vec = self.get_vectorized()
        next_s_vec_list = [self.get_next_state(s_vec, a) for a in a_vec]
        P = np.stack([self.get_ind_from_state(next_s_vec).T for next_s_vec in next_s_vec_list], 0)

        if verbose:
            M = 10 ** 3
            high = np.array([self.max_angle, 1])
            test_s = np.random.uniform(low=-high, high=high, size=[M, 2])
            next_s = np.stack([self.get_next_state(test_s, a) for a in a_vec], 0)
            test_s_ind = self.get_ind_from_state(test_s).T

            err = 0.
            for a_ind in range(self.num_actions):
                next_s_pred = s_vec.T @ P[a_ind] @ test_s_ind
                err += np.mean(np.linalg.norm(next_s[a_ind] - next_s_pred.T, axis=-1))
            err /= self.num_actions
            print(f'(h={h}) 1 step prediction error: {err:.4f}')

        return P

    def get_reward_mat(self):
        s_vec, a_vec = self.get_vectorized()
        R = []
        for a in a_vec:
            R.append(self.get_reward(s_vec, a).flatten())
        return np.stack(R)

    def render(self, mode="human"):
        # adapted from gym/envs/classic_control/pendulum.py
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            self.rod = rendering.make_capsule(1, 0.2)
            self.rod.set_color(0.3, 0.8, 0.3)
            self.pole_transform = rendering.Transform()
            self.rod.add_attr(self.pole_transform)
            self.viewer.add_geom(self.rod)
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

        if self.state[0] <= self.min_angle[0] or self.state[0] >= self.max_angle[0]:
            self.rod.set_color(0.8, 0.3, 0.3)
        else:
            self.rod.set_color(0.3, 0.8, 0.3)
        if self.last_a:
            torque = self.last_a.T
            self.imgtrans.scale = (- torque / 2, np.abs(torque) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        # adapted from gym/envs/classic_control/pendulum.py
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class DiscretizedPendulumDet(DiscretizedPendulum):
    def __init__(self, N, NT):
        super(DiscretizedPendulumDet, self).__init__(N=N, NT=NT)
        self.init_state, _ = super(DiscretizedPendulumDet, self).get_vectorized()
        self.init_state = self.init_state[0:len(self.init_state):10]
        self.n = 0

    def reset(self):
        self.set_state(self.init_state[self.n])
        self.n = (self.n + 1) % len(self.init_state)
        self.last_a = None
        return self._get_obs()

    def get_init_vector(self):
        s_vec = deepcopy(self.init_state)
        a_vec = np.arange(self.num_actions)
        return s_vec, a_vec


def angle_normalize(x):
    """ 각 x 가 -pi ~ pi 사이 값으로 표현되도록 변환 """
    return ((x + np.pi) % (2 * np.pi)) - np.pi
