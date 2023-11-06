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

    def __init__(self, obs_info, acts_info):
        super(DiscretizedPendulum, self).__init__(obs_info, acts_info)
        self.dt = 0.025
        self.g = 9.81
        self.m = 17.2955
        self.l = 0.7970
        self.lc = 0.5084
        self.I = 0.878121 + self.m * self.lc**2
        self.Q = np.diag([2.8139, 1.04872182])
        self.R = np.diag([1.617065e-4 * 40**2])

        self.max_angle = None
        self.max_speed = None
        self.min_angle = None
        self.min_speed = None
        self.max_torques = None
        self.min_torques = None

        self.state = None
        self.viewer = None
        self.last_a = None

        self.set_bounds(
            max_states=obs_info.high,
            min_states=obs_info.low,
            max_torques=acts_info.high,
            min_torques=acts_info.low,
        )

    def reset(self):
        high = np.array([0.025, 0.15])
        low = np.array([-0.025, -0.03])
        # high = np.array([*self.max_angle - 0.01, *self.max_speed - 0.03])
        # low = np.array([*self.min_angle + 0.01, *self.min_speed + 0.03])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs()

    def step(self, action: np.ndarray):
        assert self.state is not None, "Can't step the environment before calling reset function"
        assert action in self.action_space, f"{action} is Out of action space"
        info = {'obs': self.state.reshape(1, -1), 'act': action.reshape(1, -1)}
        r = self.get_reward(self.state, action)
        self.state = self.get_next_state(self.state, action)
        self.last_a = action
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

        self.obs_high = np.array(max_states)
        self.obs_low = np.array(min_states)

        self.obs_info.set_info(self.obs_high, self.obs_low)
        self.acts_info.set_info(max_torques, min_torques)
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float64)
        self.action_space = gym.spaces.Box(low=min_torques, high=max_torques, dtype=np.float64)

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
        new_th = th + thd * self.dt

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
        tot_dims = self.obs_info.num_cells
        tot_idx = self.get_idx_from_obs(state)

        if state_backup.ndim == 1:
            ind_vec = np.zeros(tot_dims).astype('i')
            ind_vec[tot_idx.ravel()] = 1
        else:
            batch_size = state.shape[0]
            ind_vec = csc_matrix((np.ones(batch_size), (np.arange(batch_size), tot_idx.ravel())),
                                 shape=[batch_size, tot_dims])

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
            for a_ind in range(self.acts_info.num_cells):
                next_s_pred = s_vec.T @ P[a_ind] @ test_s_ind
                err += np.mean(np.linalg.norm(next_s[a_ind] - next_s_pred.T, axis=-1))
            err /= self.acts_info.num_cells
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
    def __init__(self, obs_info, acts_info, init_states=None):
        super(DiscretizedPendulumDet, self).__init__(obs_info, acts_info)
        self._init_states = init_states
        self.n = 0

    @property
    def init_states(self):
        if self._init_states is None:
            init_states, _ = self.get_vectorized()
            self._init_states = init_states[0:len(init_states):50]
        self._init_states = np.array(self._init_states)
        return self._init_states

    def reset(self):
        self.set_state(self.init_states[self.n])
        self.n = (self.n + 1) % len(self.init_states)
        self.last_a = None
        return self._get_obs()

    def get_init_vector(self):
        s_vec = deepcopy(self.init_states)
        a_vec = self.acts_info.get_vectorized()
        return s_vec, a_vec


def angle_normalize(x):
    """ 각 x 가 -pi ~ pi 사이 값으로 표현되도록 변환 """
    return ((x + np.pi) % (2 * np.pi)) - np.pi
