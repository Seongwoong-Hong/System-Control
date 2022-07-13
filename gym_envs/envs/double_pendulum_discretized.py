from os import path

import gym
import numpy as np
from copy import deepcopy
from scipy.sparse import csc_matrix

from gym_envs.envs.utils import calc_trans_mat_error, angle_normalize
from gym_envs.envs import BaseDiscEnv, UncorrDiscreteInfo


class DiscretizedDoublePendulum(BaseDiscEnv):
    """
    State, action 이 이산화된 Double Pendulum 환경, 전환 행렬 P 계산
    Reward 는 state 에 대한 2 차 cost 의 음수임
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, N=None, NT=np.array([11, 11])):
        super(DiscretizedDoublePendulum, self).__init__()
        self.dt = 0.025
        self.g = 9.81
        self.Is = [0.1, 0.1]
        self.ms = [1., 1.]
        self.lcs = [0.5, 0.5]
        self.ls = [1., 1.]
        # self.Q = np.diag([3.1139, 1.2872182, 0.14639979, 0.04540204])
        # self.Q = np.diag([3.5139, 1.2872182, 0.14639979, 0.10540204])
        self.Q = np.diag([3.5139, 0.2872182, 0.24639979, 0.01540204])
        # self.Q = np.diag([1.5139, 1.0872182, 0.7639979, 0.4540204])
        # self.R = np.diag([0.02537065, 0.01358577])
        self.R = np.diag([0.02537065*2500/1600, 0.01358577*(65**2)/900])

        self.max_angles = None
        self.max_speeds = None
        self.min_angles = None
        self.min_speeds = None

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
        self.num_actions = NT
        self.obs_info = UncorrDiscreteInfo(self.num_cells)
        self.acts_info = UncorrDiscreteInfo(self.num_actions)
        self.set_bounds(
            max_states=[0.05, 0.05, 0.3, 0.35],
            min_states=[-0.05, -0.2, -0.08, -0.4],
            # max_torques=[40., 30.],
            max_torques=[50., 65.],
            # min_torques=[-20., -10.,],
            min_torques=[-30., -10.,],
        )

    def reset(self):
        high = np.array([*self.max_angles, *self.max_speeds])
        low = np.array([*self.min_angles, *self.min_speeds])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs()

    def step(self, action: np.ndarray):
        assert self.state is not None, "Can't step the environment before calling reset function"
        assert action in self.action_space, f"{action} is Out of action space"
        self.last_a = action
        info = {'obs': self.state.reshape(1, -1), 'acts': action.reshape(1, -1)}
        r = self.get_reward(self.state, action)
        self.state = self.get_next_state(self.state[None, ...], action[None, ...])[0, 0, ...]

        return self._get_obs(), r, False, info

    def set_state(self, state):
        assert state in self.observation_space
        self.state = state

    def set_bounds(self, max_states, min_states, max_torques, min_torques):
        self.max_angles = np.array(max_states[:2])
        self.max_speeds = np.array(max_states[2:])
        self.min_angles = np.array(min_states[:2])
        self.min_speeds = np.array(min_states[2:])

        self.obs_high = np.array([*self.max_angles, *self.max_speeds])
        self.obs_low = np.array([*self.min_angles, *self.min_speeds])
        self.max_torques = np.array(max_torques)
        self.min_torques = np.array(min_torques)

        self.obs_info.set_info(self.obs_high, self.obs_low)
        self.acts_info.set_info(self.max_torques, self.min_torques)
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
        th0, th1, thd0, thd1 = np.split(np.copy(state), (1, 2, 3), axis=-1)
        T0, T1 = action.T[..., None, None]
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

        # 충돌 시 upper rod eom 을 single pendulum 으로 변경
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
        thd1[c_ind] = thd1[c_ind] + (
                T1[c_ind] + m1 * g * lc1 * np.sin(th0_before[c_ind] + th1[c_ind])) / A11[c_ind] * dt
        th1 = th1 + thd1 * dt

        # 완전 비탄성 충돌
        thd0[np.logical_or(self.max_angles[0] <= th0, th0 <= self.min_angles[0])] = 0.
        thd1[np.logical_or(self.max_angles[1] <= th1, th1 <= self.min_angles[1])] = 0.

        th0 = np.clip(th0, self.min_angles[0], self.max_angles[0])
        th1 = np.clip(th1, self.min_angles[1], self.max_angles[1])
        thd0 = np.clip(thd0, self.min_speeds[0], self.max_speeds[0])
        thd1 = np.clip(thd1, self.min_speeds[1], self.max_speeds[1])

        return np.concatenate([th0, th1, thd0, thd1], axis=-1)

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
            ind_vec[tot_idx] = 1
        else:
            batch_size = state.shape[0]
            ind_vec = csc_matrix((np.ones(batch_size), (np.arange(batch_size), tot_idx)),
                                 shape=[batch_size, np.prod(dims)])

        return ind_vec

    def get_trans_mat(self, h=None, verbose=False):
        # Transition Matrix shape: (|A|, |Next S|, |S|)
        s_vec, a_vec = self.get_vectorized()
        next_s_vec_list = self.get_next_state(s_vec, a_vec)

        P = np.stack([self.get_ind_from_state(next_s_vec).T for next_s_vec in next_s_vec_list], 0)

        if verbose:
            high = np.array([*self.max_angles, *self.max_speeds])
            low = np.array([*self.min_angles, *self.min_speeds])
            err = calc_trans_mat_error(self, s_vec, a_vec, np.random.uniform(low=low, high=high, size=[1000, 4]),
                                       P) / high[None, :]
            print(f'1 step prediction error: {err.mean(axis=0)}')

        return P

    def get_reward_mat(self):
        s_vec, a_vec = self.get_vectorized()
        R = []
        for a in a_vec:
            R.append(self.get_reward(s_vec, a).flatten())
        return np.stack(R)

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            height = sum(self.ls)
            self.viewer = rendering.Viewer(500, 700)
            self.viewer.set_bounds(- 0.5 * height, 0.5 * height, -0.2 * height, 1.2 * height)

            self.rod = rendering.make_capsule(self.ls[0], 0.2)
            self.rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform0 = rendering.Transform()
            self.rod.add_attr(self.pole_transform0)
            self.viewer.add_geom(self.rod)
            axle0 = rendering.make_circle(0.05)
            axle0.set_color(0, 0, 0)
            self.viewer.add_geom(axle0)

            self.rod1 = rendering.make_capsule(self.ls[1], 0.2)
            self.rod1.set_color(0.8, 0.3, 0.3)
            self.pole_transform1 = rendering.Transform()
            self.rod1.add_attr(self.pole_transform1)
            self.viewer.add_geom(self.rod1)
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

        if self.state[0] <= self.min_angles[0] or self.state[0] >= self.max_angles[0]:
            self.rod.set_color(0.8, 0.3, 0.3)
        else:
            self.rod.set_color(0.3, 0.8, 0.3)
        if self.state[1] <= self.min_angles[1] or self.state[1] >= self.max_angles[1]:
            self.rod1.set_color(0.8, 0.3, 0.3)
        else:
            self.rod1.set_color(0.3, 0.8, 0.3)

        if self.last_a is not None:
            torques = self.last_a.T
            self.imgtrans0.scale = (- torques[0] / self.max_torques[0], np.abs(torques[0]) / self.max_torques[0])
            self.imgtrans1.scale = (- torques[1] / self.max_torques[1], np.abs(torques[1]) / self.max_torques[1])
            self.imgtrans1.set_translation(hinge_x, hinge_y)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class DiscretizedDoublePendulumDet(DiscretizedDoublePendulum):
    def __init__(self, N=None, init_states=None, NT=np.array([11, 11])):
        super(DiscretizedDoublePendulumDet, self).__init__(N=N, NT=NT)
        self._init_states = init_states
        self.n = 0

    @property
    def init_states(self):
        if self._init_states is None:
            init_states, _ = self.get_vectorized()
            return init_states[0:len(init_states):1000]
        else:
            return np.array(self._init_states).reshape(-1, 4)

    def reset(self):
        self.set_state(self.init_states[self.n])
        self.n = (self.n + 1) % len(self.init_states)
        self.last_a = None
        return self._get_obs()

    def get_init_vector(self):
        s_vec = deepcopy(self.init_states)
        a_vec = self.acts_info.get_vectorized()
        return s_vec, a_vec


class DiscretizedHuman(DiscretizedDoublePendulum):
    def __init__(self, bsp=None, N=None, NT=np.array([19, 19])):
        super(DiscretizedHuman, self).__init__(N=N, NT=NT)
        if bsp is not None:
            m_u, l_u, com_u, I_u = bsp[6, :]
            m_s, l_s, com_s, I_s = bsp[2, :]
            m_t, l_t, com_t, I_t = bsp[3, :]
            l_l = l_s + l_t
            m_l = 2 * (m_s + m_t)
            com_l = (m_s * com_s + m_t * (l_s + com_t)) / (m_s + m_t)
            I_l = 2 * (I_s + m_s * (com_l - com_s) ** 2 + I_t + m_t * (com_l - (l_s + com_t)) ** 2)
            self.Is = [I_l, I_u]
            self.ms = [m_l, m_u]
            self.lcs = [com_l, com_u]
            self.ls = [l_l, l_u]


class DiscretizedHumanDet(DiscretizedDoublePendulumDet):
    def __init__(self, bsp=None, N=None, init_states=None, NT=np.array([19, 19])):
        super(DiscretizedHumanDet, self).__init__(N=N, init_states=init_states, NT=NT)
        if bsp is not None:
            m_u, l_u, com_u, I_u = bsp[6, :]
            m_s, l_s, com_s, I_s = bsp[2, :]
            m_t, l_t, com_t, I_t = bsp[3, :]
            l_l = l_s + l_t
            m_l = 2 * (m_s + m_t)
            com_l = (m_s * com_s + m_t * (l_s + com_t)) / (m_s + m_t)
            I_l = 2 * (I_s + m_s * (com_l - com_s) ** 2 + I_t + m_t * (com_l - (l_s + com_t)) ** 2)
            self.Is = [I_l, I_u]
            self.ms = [m_l, m_u]
            self.lcs = [com_l, com_u]
            self.ls = [l_l, l_u]
