"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartPoleContEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation               Min             Max
        0	Cart Velocity             -Inf            Inf
        1	Cart Position             -5              5
        3	Pole Velocity At Tip      -Inf            Inf
        4	Pole Angle                -4 rad          4 rad

    Actions:
        Type: Continuous(1,1)

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is Simliar to LQR cost function

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        1000 timesteps
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, max_ep=1000, high=np.pi/6, low=0):
        self.gravity = 9.81
        self.masscart = 5.0
        self.masspole = 1.0
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.tau = 0.005  # seconds between state updates
        self.Q = np.diag([1, 1, 1, 1])
        self.R = 0.001
        self.kinematics_integrator = 'euler'
        self.max_ep = max_ep

        # Angle at which to fail the episode
        self.theta_threshold_radians = 4
        self.x_threshold = 5
        self.high = high
        self.low = low
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Box(shape=(1,), high=2000, low=-2000)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

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

        x_dot, x, theta_dot, theta = self.state
        force = action.squeeze()
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) /\
                   (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.set_state([x_dot, x, theta_dot, theta])

        reward = 1.025 - 0.01 * (self.state.T @ self.Q @ self.state + force * self.R * force)
        # torch/ppo_ct l_1,2 use this reward
        # tf/ppo_ctl_5,6 second learn use this reward

        # reward = np.exp(-0.25 * (self.state.T @ self.Q @ self.state + action * self.R * action))
        # tf/ppo_ctl_5,6 use this reward

        self.traj_len += 1
        done = False
        if self.traj_len == self.max_ep:
            done = True
            self.traj_len = 0
        elif theta > 2*self.high or theta < -2*self.high or x > self.x_threshold+2.5 or x < -self.x_threshold-2.5:
            done = True
            reward -= 2 * (self.max_ep - self.traj_len)
            self.traj_len = 0

        return self.state.squeeze(), reward[0], done, {'action': action}

    def reset(self):
        if np.random.uniform() < 0.5:
            self.state = self.np_random.uniform(low=self.low, high=self.high, size=(4,))
        else:
            self.state = self.np_random.uniform(low=-self.high, high=-self.low, size=(4,))
        self.state[0], self.state[1] = 0, 0
        self.steps_beyond_done = None
        self.traj_len = 0
        return self.state.squeeze()

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 5/60 * scale * 2
        polelen = scale * (2 * self.length) * 2
        cartwidth = 25/60 * scale * 2
        cartheight = 15/60 * scale * 2

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
            self.axle = rendering.make_circle(polewidth/2)
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
        self.state = np.concatenate((np.array([state[0]]), np.array([state[1]]),
                                     np.array([state[2]]), np.array([state[3]]))).squeeze()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
