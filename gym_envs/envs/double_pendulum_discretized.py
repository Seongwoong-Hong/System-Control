from os import path

import gym
import numpy as np
from copy import deepcopy
from gym.utils import seeding
from scipy.sparse import csc_matrix

from gym_envs.envs.utils import calc_trans_mat_error, angle_normalize


class DiscretizedDoublePendulum(gym.Env):
    """
    State, action 이 이산화된 Double Pendulum 환경, 전환 행렬 P 계산
    Reward 는 state 에 대한 2 차 cost 의 음수임
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, h=None):
        super(DiscretizedDoublePendulum, self).__init__()
        self.max_torques = [20., 10.]
        self.max_speeds = [1., 1.]
        self.max_angles = [np.pi / 3, np.pi / 6]
        self.dt = 0.05
        self.g = 9.81
        self.Is = [0.1, 0.1]
        self.ms = [1., 1.]
        self.lcs = [0.5, 0.5]
        self.ls = [1., 1.]
        self.h = h
        self.num_actions = [5, 5]
        self.Q = np.diag([1., 1., 0., 0.])

        self.np_random = None
        self.state = None
        self.viewer = None
        self.last_a = None

        obs_high = np.array([*self.max_angles, *self.max_speeds]) + h
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(self.num_actions)
        self.torque_lists = [np.linspace(-max_t, max_t, n_act)
                             for max_t, n_act in zip(self.max_torques, self.num_actions)]

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random()
        return [seed]

    def reset(self):
        high = np.array([*self.max_angles, 1., 1.])
        self.state = self.np_random.uniform(low=-high, high=high)

        return self.get_obs()

    def step(self, action: np.ndarray):
        assert self.state is not None, "Can't step the environment before calling reset function"
        assert action in self.action_space, f"{action} is Out of action space"
        self.last_a = action
        info = {'obs': self.state.reshape(1, -1), 'act': action.reshape(1, -1)}
        r = self.get_reward(self.state, action)
        self.state = self.get_next_state(self.state[None, ...], action[None, ...])[0, 0, ...]

        return self.get_obs(), r, False, info

    def set_state(self, state):
        assert state in self.observation_space
        self.state = state

    def get_torque(self, actions):
        a0, a1 = np.split(actions, 2, axis=-1)
        t0_list, t1_list = self.torque_lists

        return t0_list[a0], t1_list[a1]

    def get_reward(self, state, action):
        if state.ndim == 1:
            state[0] = angle_normalize(state[0])
            state[1] = angle_normalize(state[1])
            r = - np.sum((state[None, ...] @ self.Q) * state, axis=-1)
        else:
            state[:, 0] = angle_normalize(state[:, 0])
            state[:, 1] = angle_normalize(state[:, 1])
            r = - np.sum((state @ self.Q) * state, axis=-1)

        return r

    def get_next_state(self, state, action):
        th0, th1, thd0, thd1 = np.split(np.copy(state), (1, 2, 3), axis=-1)
        T0, T1 = self.get_torque(np.expand_dims(action, axis=1))
        g, (I0, I1), (m0, m1), (lc0, lc1), (l0, l1), dt = \
            self.g, self.Is, self.ms, self.lcs, self.ls, self.dt

        A00 = I0 + m0 * lc0 ** 2 + I1 + m1 * l0 ** 2 + 2 * m1 * l0 * lc1 * np.cos(th1) + m1 * lc1 ** 2
        A01 = I1 + m1 * l0 * lc1 * np.cos(th1) + m1 * lc1 ** 2
        A10 = I1 + m1 * lc1 ** 2 + m1 * l0 * lc1 * np.cos(th1)
        A11 = np.array(I1 + m1 * lc1 ** 2) * np.ones_like(th0)

        b0 = T0 + m1 * l0 * lc1 * thd1 * (2 * thd0 + thd1) * np.sin(th1) + \
             (m0 * lc0 + m1 * l0) * g * np.sin(th0) + m1 * g * lc1 * np.sin(th0 + th1)
        b1 = T1 - m1 * l0 * lc1 * thd0 ** 2 * np.sin(th1) + m1 * g * lc1 * np.sin(th0 + th1)

        A_det = A00 * A11 - A01 * A10
        thd0 = thd0 + (A11 * b0 - A01 * b1) / A_det * dt
        th0_before = np.copy(th0)
        th0 = th0 + thd0 * dt

        # 충동 시 upper rod eom 을 single pendulum 으로 변경
        c_ind = np.logical_and(np.abs(th0_before) >= self.max_angles[0] - 1e-3, th0 * thd0 > 0)
        nc_ind = np.logical_not(c_ind)

        th0_before = np.tile(th0_before, (action.shape[0], 1, 1))
        th1 = np.tile(th1, (action.shape[0], 1, 1))
        thd1 = np.tile(thd1, (action.shape[0], 1, 1))
        A00 = np.tile(A00, (action.shape[0], 1, 1))
        A10 = np.tile(A10, (action.shape[0], 1, 1))
        A11 = np.tile(A11, (action.shape[0], 1, 1))
        A_det = np.tile(A_det, (action.shape[0], 1, 1))
        T1 = np.tile(T1, (1, state.shape[0], 1))

        thd1[nc_ind] = thd1[nc_ind] + (-A10[nc_ind] * b0[nc_ind] + A00[nc_ind] * b1[nc_ind]) / A_det[nc_ind] * dt
        thd1[c_ind] = thd1[c_ind] + \
                       (T1[c_ind] + m1 * g * lc1 * np.sin(th0_before[c_ind] + th1[c_ind])) / A11[c_ind] * dt
        th1 = th1 + thd1 * dt

        # 완전 비탄성 충돌
        thd0[np.logical_or(self.max_angles[0] <= th0, th0 <= -self.max_angles[0])] = 0.
        thd1[np.logical_or(self.max_angles[1] <= th1, th1 <= -self.max_angles[1])] = 0.

        th0 = np.clip(th0, -self.max_angles[0], self.max_angles[0])
        th1 = np.clip(th1, -self.max_angles[1], self.max_angles[1])
        thd0 = np.clip(thd0, -self.max_speeds[0], self.max_speeds[0])
        thd1 = np.clip(thd1, -self.max_speeds[1], self.max_speeds[1])

        return np.concatenate([th0, th1, thd0, thd1], axis=-1)

    def get_obs(self):
        return self.state

    def get_num_cells(self, h=None):
        if h is None:
            h = self.h
        assert h is not None
        h_th0, h_th1, h_thd0, h_thd1 = h
        n_th0, n_th1 = np.round(2 * (self.max_angles / np.array([h_th0, h_th1]))).astype('i') + 1
        n_thd0, n_thd1 = np.round(2 * (self.max_speeds / np.array([h_thd0, h_thd1]))).astype('i') + 1

        return n_th0, n_th1, n_thd0, n_thd1

    def get_vectorized(self, h=None):
        if h is None:
            h = self.h
        assert h is not None
        h_th0, h_th1, h_thd0, h_thd1 = h
        n_th0, n_th1, n_thd0, n_thd1 = self.get_num_cells(h)

        s_vec = np.stack(np.meshgrid(h_th0 * np.arange(n_th0) - self.max_angles[0],
                                     h_th1 * np.arange(n_th1) - self.max_angles[1],
                                     h_thd0 * np.arange(n_thd0) - self.max_speeds[0],
                                     h_thd1 * np.arange(n_thd1) - self.max_speeds[1],
                                     indexing='ij'),
                         -1).reshape(-1, 4)
        a_vec = np.stack(np.meshgrid(np.arange(self.num_actions[0]),
                                     np.arange(self.num_actions[1]),
                                     indexing='ij'),
                         -1).reshape(-1, 2)

        return s_vec, a_vec

    def get_ind_from_state(self, state, h=None):
        if h is None:
            h = self.h
        assert h is not None
        dims = self.get_num_cells(h)

        state_sub = np.round((state + np.concatenate([self.max_angles, self.max_speeds])) / np.array(h)).astype('i')
        tot_idx = np.ravel_multi_index(state_sub.T, dims, order='C')

        if state.ndim == 1:
            ind_vec = np.zeros(np.prod(dims)).astype('i')
            ind_vec[tot_idx.ravel()] = 1
        else:
            batch_size = state.shape[0]
            ind_vec = csc_matrix((np.ones(batch_size), (np.arange(batch_size), tot_idx.astype('i'))),
                                 shape=[batch_size, np.prod(dims)])

        return ind_vec

    def get_idx_from_obs(self, obs: np.ndarray):
        dims = self.get_num_cells(self.h)

        obs_sub = np.round((obs + np.concatenate([self.max_angles, self.max_speeds])) / np.array(self.h)).astype('i')
        tot_idx = np.ravel_multi_index(obs_sub.T, dims, order='C')
        return tot_idx.flatten()

    def get_act_from_idx(self, idx: np.ndarray):
        a_vec = np.stack(np.meshgrid(np.arange(self.num_actions[0]),
                                     np.arange(self.num_actions[1]),
                                     indexing='ij'),
                         -1).reshape(-1, 2)
        return a_vec[idx.flatten()]

    def get_trans_mat(self, h=None, verbose=False):
        if h is None:
            h = self.h
        assert h is not None
        s_vec, a_vec = self.get_vectorized(h)
        next_s_vec_list = self.get_next_state(s_vec, a_vec)

        P = np.stack([self.get_ind_from_state(next_s_vec, h).T for next_s_vec in next_s_vec_list], 0)

        if verbose:
            high = np.array([*self.max_angles, 1., 1.])
            err = calc_trans_mat_error(self, P, s_vec, a_vec, h, 10 ** 3,
                                       sampler=lambda m: np.random.uniform(low=-high, high=high, size=[m, 4]))
            print(f'(h={h}) 1 step prediction error: {err:.4f}')

        return P

    def get_action_mat(self, pi, h=None):
        if h is None:
            h = self.h
        assert h is not None
        s_vec, _ = self.get_vectorized(h)
        return pi(s_vec)

    def get_reward_vec(self, h=None):
        if h is None:
            h = self.h
        assert h is not None
        s_vec, _ = self.get_vectorized(h)
        R = self.get_reward(s_vec, None)
        return R

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            height = sum(self.ls)
            self.viewer = rendering.Viewer(500, 700)
            self.viewer.set_bounds(- 0.5 * height, 0.5 * height, -0.2 * height, 1.2 * height)

            rod0 = rendering.make_capsule(self.ls[0], 0.2)
            rod0.set_color(0.8, 0.3, 0.3)
            self.pole_transform0 = rendering.Transform()
            rod0.add_attr(self.pole_transform0)
            self.viewer.add_geom(rod0)
            axle0 = rendering.make_circle(0.05)
            axle0.set_color(0, 0, 0)
            self.viewer.add_geom(axle0)

            rod1 = rendering.make_capsule(self.ls[1], 0.2)
            rod1.set_color(0.8, 0.3, 0.3)
            self.pole_transform1 = rendering.Transform()
            rod1.add_attr(self.pole_transform1)
            self.viewer.add_geom(rod1)
            axle1 = rendering.make_circle(0.05)
            axle1.set_color(0, 0, 0)
            self.axle_transform1 = rendering.Transform()
            axle1.add_attr(self.axle_transform1)
            self.viewer.add_geom(axle1)

            fname = path.join(path.dirname(gym.__file__), 'envs',
                              'classic_control', 'assets', 'clockwise.png')
            self.img0 = rendering.Image(fname, 0.5, 0.5)
            self.imgtrans0 = rendering.Transform()
            self.img0.add_attr(self.imgtrans0)
            self.img1 = rendering.Image(fname, 0.5, 0.5)
            self.imgtrans1 = rendering.Transform()
            self.img1.add_attr(self.imgtrans1)

        self.viewer.add_onetime(self.img0)
        self.viewer.add_onetime(self.img1)
        self.pole_transform0.set_rotation(self.state[0] + np.pi / 2)
        self.pole_transform1.set_rotation(self.state[0] + self.state[1] + np.pi / 2)
        hinge_x = self.ls[0] * np.cos(self.state[0] + np.pi / 2)
        hinge_y = self.ls[0] * np.sin(self.state[0] + np.pi / 2)
        self.pole_transform1.set_translation(hinge_x, hinge_y)
        self.axle_transform1.set_translation(hinge_x, hinge_y)

        if self.last_a is not None:
            torques = self.get_torque(self.last_a)
            self.imgtrans0.scale = (- torques[0] / self.max_torques[0], np.abs(torques[0]) / self.max_torques[0])
            self.imgtrans1.scale = (- torques[1] / self.max_torques[1], np.abs(torques[1]) / self.max_torques[1])
            self.imgtrans1.set_translation(hinge_x, hinge_y)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class DiscretizedDoublePendulumDet(DiscretizedDoublePendulum):
    def __init__(self, h=None):
        super(DiscretizedDoublePendulumDet, self).__init__(h=h)
        self.init_state, _ = super(DiscretizedDoublePendulumDet, self).get_vectorized()
        self.init_state = self.init_state[0:len(self.init_state):100]
        self.n = 0

    def reset(self):
        self.set_state(self.init_state[self.n])
        self.n = (self.n + 1) % len(self.init_state)
        self.last_a = None
        return self.get_obs()

    def get_vectorized(self, h=None):
        s_vec = deepcopy(self.init_state)
        a_vec = np.stack(np.meshgrid(np.arange(self.num_actions[0]),
                                     np.arange(self.num_actions[1]),
                                     indexing='ij'),
                         -1).reshape(-1, 2)
        return s_vec, a_vec
