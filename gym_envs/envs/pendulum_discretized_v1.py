import gym
import numpy as np

from .pendulum_discretized import DiscretizedPendulum


class DiscretizedPendulumV1(DiscretizedPendulum):
    def __init__(self, obs_info, acts_info):
        self.max_acc = None
        self.min_acc = None
        super(DiscretizedPendulumV1, self).__init__(obs_info, acts_info)
        self.Q = np.diag([2.8139, 1.04872182, 0.0])

    def reset(self):
        high = np.array([*self.max_angle - 0.01, *self.max_speed - 0.03, 2.])
        low = np.array([*self.min_angle + 0.01, *self.min_speed + 0.03, -2.])
        self.state = self.np_random.uniform(low=low, high=high)
        return self._get_obs()

    def set_bounds(self, max_states, min_states, max_torques, min_torques):
        self.max_angle = np.array(max_states)[:1]
        self.max_speed = np.array(max_states)[1:2]
        self.max_acc = np.array(max_states)[2:]
        self.min_angle = np.array(min_states)[:1]
        self.min_speed = np.array(min_states)[1:2]
        self.min_acc = np.array(min_states)[2:]
        self.max_torques = np.array(max_torques)
        self.min_torques = np.array(min_torques)

        self.obs_high = np.array(max_states)
        self.obs_low = np.array(min_states)

        self.obs_info.set_info(self.obs_high, self.obs_low)
        self.acts_info.set_info(max_torques, min_torques)
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float64)
        self.action_space = gym.spaces.Box(low=min_torques, high=max_torques, dtype=np.float64)

    def get_next_state(self, state, action):
        th, thd, thdd = np.split(state, 3, axis=-1)
        torque = action.T
        torque = torque[..., None] if state.ndim == 2 else torque

        new_thdd = ((self.m * self.g * self.lc) / self.I * np.sin(th) +  torque / self.I)
        new_thd = thd + new_thdd * self.dt
        new_th = th + thd * self.dt

        new_th = np.clip(new_th, self.min_angle, self.max_angle)
        new_thd = np.clip(new_thd, self.min_speed, self.max_speed)
        new_thdd = np.clip(new_thdd, self.min_acc, self.max_acc)

        if state.ndim == 1:
            return np.array([new_th[0], new_thd[0], new_thdd[0]])
        else:
            return np.column_stack([new_th, new_thd, new_thdd])