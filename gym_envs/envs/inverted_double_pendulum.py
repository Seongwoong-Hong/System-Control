"""
inverted double pendulum system, modified from the cartpole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class InvertedDoublePendulum(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.g = 9.81
        self.m1, self.m2 = 5.0, 5.0
        self.l1, self.l2 = 1.0, 1.0
        self.com1, self.com2 = 0.5, 0.5
        self.I1, self.I2 = 0.41667, 0.41667
        self.action_coeff = 200
        self.dt = 0.005  # seconds between state updates
        self.Q = np.diag([1, 1, 1, 1])
        self.R = 0.001
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.angle_limits = 0.5
        self.angvel_limits = 2.0
        self.high = np.array([self.angle_limits,
                              self.angle_limits,
                              self.angvel_limits,
                              self.angvel_limits],
                             dtype=np.float64)

        self.action_space = spaces.Box(shape=(2,), high=1, low=-1)
        self.observation_space = spaces.Box(-self.high, self.high, dtype=np.float64)

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.traj_len = 0

        self.np_random = None

        # parameters for render
        self.carttrans = None
        self.poletrans = None
        self.axle = None
        self.track = None
        self._pole_geom = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        th1, th2, dth1, dth2 = self.state
        reward = - abs(th1) - abs(th2)
        T1, T2 = action * self.action_coeff

        A11 = self.I1 + self.m1 * self.com1 ** 2 + self.m2 * self.l1 ** 2
        A12 = self.m2 * self.l1 * self.com2 * math.cos(th1 - th2)
        A21 = self.m2 * self.l1 * self.com2 * math.cos(th1 - th2)
        A22 = self.I2 + self.m2 * self.com2 ** 2
        b1 = T1 - T2 - self.m2 * dth2 ** 2 * self.l1 * self.com2 * math.sin(th1 - th2) + \
            self.m1 * self.g * self.com1 * math.sin(th1) + self.m2 * self.g * self.l1 * math.sin(th1)
        b2 = T2 + self.m2 * dth1 * dth1 * self.l1 * self.com2 * math.sin(
            th1 - th2) + self.m2 * self.g * self.com2 * math.sin(th2)

        A = np.array([[A11, A12],
                      [A21, A22]])
        b = np.array([[b1],
                      [b2]])

        ddth1, ddth2 = np.linalg.inv(A) @ b

        if self.kinematics_integrator == 'euler':
            th1 = th1 + self.dt * dth1
            th2 = th2 + self.dt * dth2
            dth1 = dth1 + self.dt * ddth1.item()
            dth2 = dth2 + self.dt * ddth2.item()
        else:  # semi-implicit euler
            raise NotImplementedError

        self.set_state([th1, th2, dth1, dth2])

        done = False

        if abs(th1) > self.angle_limits or \
                abs(th1) > self.angle_limits:
            done = True

        return self.state, reward, done, {'action': action}

    def reset(self):
        self.set_state(self.np_random.uniform(low=-self.high, high=self.high))
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        scale = 1.5
        polewidth = scale * 0.1
        pole1len = scale * self.l1
        pole2len = scale * self.l2

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-3, 3, -1, 3)
            l, r, t, b = -polewidth / 2, polewidth / 2, pole1len, 0
            pole1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole1.set_color(.8, .6, .4)
            self.pole1_trans = rendering.Transform()
            pole1.add_attr(self.pole1_trans)
            self.viewer.add_geom(pole1)
            self.axle1 = rendering.make_circle(polewidth / 2)
            self.axle1.add_attr(self.pole1_trans)
            self.axle1.set_color(0, 0, 0)
            self.viewer.add_geom(self.axle1)
            l, r, t, b = -polewidth / 2, polewidth / 2, pole2len, 0
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2.set_color(.8, .6, .4)
            self.pole2_trans = rendering.Transform(translation=(0, pole1len))
            pole2.add_attr(self.pole2_trans)
            pole2.add_attr(self.pole1_trans)
            self.viewer.add_geom(pole2)
            self.axle2 = rendering.make_circle(polewidth / 2)
            self.axle2.add_attr(self.pole2_trans)
            self.axle2.add_attr(self.pole1_trans)
            self.axle2.set_color(0, 0, 0)
            self.viewer.add_geom(self.axle2)

            self._pole1_geom = pole1
            self._pole2_geom = pole2

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole1 = self._pole1_geom
        pole2 = self._pole2_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, pole1len, 0
        pole1.v = [(l, b), (l, t), (r, t), (r, b)]
        l, r, t, b = -polewidth / 2, polewidth / 2, pole2len, 0
        pole2.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        self.pole1_trans.set_rotation(-x[0])
        self.pole2_trans.set_rotation(-x[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def set_state(self, state):
        self.state = state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
