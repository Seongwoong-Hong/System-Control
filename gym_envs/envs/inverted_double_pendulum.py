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
        self.tau = 0.005  # seconds between state updates
        self.Q = np.diag([1, 1, 1, 1])
        self.R = 0.001
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.angle_limits = 0.5
        self.angvel_limits = 0.5
        self.high = np.array([self.angle_limits,
                              self.angle_limits,
                              self.angvel_limits,
                              self.angvel_limits],
                             dtype=np.float64)

        self.action_space = spaces.Box(shape=(2,), high=1, low=-1)
        self.observation_space = spaces.Box(-self.high, self.high, dtype=np.float64)

        self.seed()
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
            th1 = th1 + self.tau * dth1
            th2 = th2 + self.tau * dth2
            dth1 = dth1 + self.tau * ddth1
            dth2 = dth2 + self.tau * ddth2
        else:  # semi-implicit euler
            raise NotImplementedError

        self.set_state([th1, th2, dth1, dth2])

        done = False

        if abs(th1) > self.angle_limits or \
                abs(th1) > self.angle_limits or \
                abs(dth1) > self.angvel_limits or \
                abs(dth2) > self.angvel_limits:
            done = True

        return self.state, reward, done, {'action': action}

    def reset(self):
        self.set_state(self.np_random.uniform(low=-self.high, high=self.high))
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        scale = 1
        polewidth = 5 / 60 * scale * 2
        polelen = scale * (2 * self.length) * 2

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[1] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[3])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def set_state(self, state):
        self.state = state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
