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

    def __init__(self, N=None, NT=np.array([11, 11])):
        super(DiscretizedDoublePendulum, self).__init__()
        self.max_torques = np.array([100., 100.])
        # self.max_speeds = np.array([2.2, 4.8])
        # self.max_angles = np.array([0.4, 1.0])
        self.max_speeds = np.array([0.8, 2.4])
        self.max_angles = np.array([0.16, 0.67])
        # self.max_speeds = np.array([1.0369735292212519, 3.110920587663755])
        # self.max_angles = np.array([0.19851002092963618, 0.8684653307227984])
        self.min_speeds = -self.max_speeds
        self.min_angles = -self.max_angles

        self.dt = 0.025
        self.g = 9.81
        self.Is = [0.1, 0.1]
        self.ms = [1., 1.]
        self.lcs = [0.5, 0.5]
        self.ls = [1., 1.]
        self.num_actions = NT
        self.Q = np.diag([0.16540, 0.14075, 0.01067, 0.00152])
        self.R = np.diag([0.00076, 0.000576])

        self.np_random = None
        self.state = None
        self.viewer = None
        self.last_a = None

        self.obs_high = np.array([*self.max_angles, *self.max_speeds])
        self.obs_low = np.array([*self.min_angles, *self.min_speeds])
        if N is int:
            assert N % 2, "N should be a odd number"
            N = N * np.ones_like(self.obs_high, dtype=int)
        else:
            N = np.array(N)
            assert N is not None, "The number of discretization should be defined"
            assert N.shape == self.obs_high.shape
            assert (N % 2).all(), "N should be consist of odd numbers"
        self.num_cells = N
        self.obs_shape = []
        for high, low, n in zip(self.obs_high, self.obs_low, self.num_cells):
            self.obs_shape.append(np.linspace(low, high, n + 1))
        self.torque_lists = []
        for high, n in zip(self.max_torques, self.num_actions):
            self.torque_lists.append(np.linspace(-high, high, n + 1))
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float64)
        self.action_space = gym.spaces.MultiDiscrete(self.num_actions)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random()
        return [seed]

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

    @property
    def disc_states(self):
        return [(os[1:] + os[:-1]) / 2 for os in self.obs_shape]

    def set_state(self, state):
        assert np.abs(state) - 1e-6 in self.observation_space
        self.state = state

    def get_torque(self, actions):
        a0, a1 = np.split(actions.astype('i'), 2, axis=-1)
        t0_list, t1_list = self.torque_lists

        return np.array([t0_list[a0], t1_list[a1]]).squeeze(axis=-1)

    def get_reward(self, state, action):
        state = deepcopy(state)
        torques = self.get_torque(action.reshape(-1, 2)).T / self.max_torques[None, ...]
        if state.ndim == 1:
            state[0] = angle_normalize(state[0])
            state[1] = angle_normalize(state[1])
            r = - np.sum((state[None, ...] @ self.Q) * state, axis=-1)
        else:
            state[:, 0] = angle_normalize(state[:, 0])
            state[:, 1] = angle_normalize(state[:, 1])
            r = - np.sum((state @ self.Q) * state, axis=-1)
        r -= np.sum((torques @ self.R) * torques, axis=-1)
        return r * 100

    def get_next_state(self, state, action):
        th0, th1, thd0, thd1 = np.split(np.copy(state), (1, 2, 3), axis=-1)
        T0, T1 = self.get_torque(action)[..., None, None]
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
        thd0[np.logical_or(self.max_angles[0] <= th0, th0 <= self.min_angles[0])] = 0.
        thd1[np.logical_or(self.max_angles[1] <= th1, th1 <= self.min_angles[1])] = 0.

        th0 = np.clip(th0, self.min_angles[0], self.max_angles[0])
        th1 = np.clip(th1, self.min_angles[1], self.max_angles[1])
        thd0 = np.clip(thd0, self.min_speeds[0], self.max_speeds[0])
        thd1 = np.clip(thd1, self.min_speeds[1], self.max_speeds[1])

        return np.concatenate([th0, th1, thd0, thd1], axis=-1)

    def _get_obs(self):
        return self.state

    def get_num_cells(self):
        return self.num_cells

    def get_vectorized(self):
        s_vec = np.stack(np.meshgrid(*self.disc_states,
                                     indexing='ij'),
                         -1).reshape(-1, 4)
        a_vec = np.stack(np.meshgrid(np.arange(self.num_actions[0]),
                                     np.arange(self.num_actions[1]),
                                     indexing='ij'),
                         -1).reshape(-1, 2)

        return s_vec, a_vec

    def get_init_vector(self):
        return self.get_vectorized()

    def get_ind_from_state(self, state, h=None):
        state_backup = deepcopy(state)
        if len(state.shape) == 1:
            state = state[None, :]
        dims = self.get_num_cells()
        tot_idx = self.get_idx_from_obs(state)

        if state_backup.ndim == 1:
            ind_vec = np.zeros(np.prod(dims)).astype('i')
            ind_vec[tot_idx] = 1
        else:
            batch_size = state.shape[0]
            ind_vec = csc_matrix((np.ones(batch_size), (np.arange(batch_size), tot_idx)),
                                 shape=[batch_size, np.prod(dims)])

        return ind_vec

    def get_idx_from_obs(self, obs: np.ndarray):
        if len(obs.shape) == 1:
            obs = obs[None, :]
        assert (np.max(obs, axis=0) <= np.append(self.max_angles, self.max_speeds) + 1e-6).all() or \
               (np.min(obs, axis=0) >= np.append(self.min_angles, self.min_speeds) - 1e-6).all()
        dims = self.get_num_cells()
        idx = []
        for i, whole_candi in enumerate(self.obs_shape):
            idx.append((obs[:, [i]] - whole_candi[:-1] >= 0).sum(axis=-1) - 1)
        tot_idx = np.ravel_multi_index(np.array(idx), dims, order='C')
        return tot_idx.flatten()

    def get_obs_from_idx(self, idx: np.ndarray):
        assert len(idx.shape) == 1
        s_vec = np.stack(np.meshgrid(*self.disc_states,
                                     indexing='ij'),
                         -1).reshape(-1, 4)
        return s_vec[idx]

    def get_acts_from_idx(self, idx: np.ndarray):
        assert len(idx.shape) == 1
        a_vec = np.stack(np.meshgrid(np.arange(self.num_actions[0]),
                                     np.arange(self.num_actions[1]),
                                     indexing='ij'),
                         -1).reshape(-1, 2)
        return a_vec[idx]

    def get_idx_from_acts(self, acts: np.ndarray):
        if len(acts.shape) == 1:
            acts = acts[None, :]
        assert (np.max(np.abs(acts), axis=0) <= self.max_torques + 1e-6).all()
        dims = np.array(self.num_actions)
        idx = []
        for i, whole_candi in enumerate(self.torque_lists):
            idx.append((acts[:, [i]] - whole_candi[:-1] >= 0).sum(axis=-1) - 1)
        tot_idx = np.ravel_multi_index(np.array(idx), dims, order='C')

        return tot_idx.flatten()

    def get_trans_mat(self, h=None, verbose=False):
        # Transition Matrix shape: (|A|, |Next S|, |S|)
        s_vec, a_vec = self.get_vectorized()
        next_s_vec_list = self.get_next_state(s_vec, a_vec)

        P = np.stack([self.get_ind_from_state(next_s_vec, h).T for next_s_vec in next_s_vec_list], 0)

        if verbose:
            high = np.array([*self.max_angles, *self.max_speeds])
            low = np.array([*self.min_angles, *self.min_speeds])
            err = calc_trans_mat_error(self, s_vec, a_vec, np.random.uniform(low=low, high=high, size=[1000, 4]),
                                       P) / high[None, :]
            print(f'1 step prediction error: {err.mean(axis=0)}')

        return P

    def get_action_mat(self, pi, h=None):
        s_vec, _ = self.get_vectorized()
        return pi(s_vec)

    def get_reward_mat(self):
        s_vec, a_vec = self.get_vectorized()
        R = []
        for a in a_vec:
            R.append(self.get_reward(s_vec, a).flatten())
        return np.stack(R)

    def get_done_mat(self):
        s_vec, a_vec = self.get_vectorized()
        d_vec = np.zeros(len(s_vec))
        d_vec[np.abs(s_vec[:, 0]) >= (self.max_angles[0] - 1e-6)] = 1
        d_vec[np.abs(s_vec[:, 1]) >= (self.max_angles[1] - 1e-6)] = 1
        return np.repeat(d_vec[None, :], len(a_vec), axis=0)

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
    def __init__(self, N=None, init_states=None, NT=np.array([11, 11])):
        super(DiscretizedDoublePendulumDet, self).__init__(N=N, NT=NT)
        if init_states is None:
            self.init_states, _ = self.get_vectorized()
            self.init_states = self.init_states[0:len(self.init_states):100]
        else:
            self.init_states = np.array(init_states).reshape(-1, 4)
        self.n = 0

    def reset(self):
        self.set_state(self.init_states[self.n])
        self.n = (self.n + 1) % len(self.init_states)
        self.last_a = None
        return self._get_obs()

    def get_init_vector(self):
        s_vec = deepcopy(self.init_states)
        a_vec = np.stack(np.meshgrid(np.arange(self.num_actions[0]),
                                     np.arange(self.num_actions[1]),
                                     indexing='ij'),
                         -1).reshape(-1, 2)
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
