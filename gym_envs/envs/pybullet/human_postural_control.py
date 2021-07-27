import os
import random
import numpy as np

from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.robot_bases import MJCFBasedRobot
from pybullet_envs.scene_abstract import SingleRobotEmptyScene


class HumanIDP(MJCFBasedRobot):
    def __init__(self, pltqs=None):
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mujoco", "assets", "HPC_custom.xml"))
        self.gear = 300
        MJCFBasedRobot.__init__(self, xml_path, 'HPC', action_dim=2, obs_dim=6)
        self._timesteps = 0
        self._pltq = None
        self._plt_torque = None
        self._pltqs = pltqs

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["hinge2"]
        self.j1.reset_current_position(0.0, 0.0)
        self.j2.reset_current_position(0.0, 0.0)
        self.j1.set_motor_torque(self.plt_torque[0])
        self.j2.set_motor_torque(self.plt_torque[1])
        return self.calc_state()

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.j1.set_motor_torque(self.gear * float(np.clip(a[0], -0.5, +0.5) + self.plt_torque[0]))
        self.j2.set_motor_torque(self.gear * float(np.clip(a[1], -0.5, +0.5) + self.plt_torque[1]))

    def calc_state(self):
        theta, theta_dot = self.j1.current_position()
        gamma, gamma_dot = self.j2.current_position()
        return np.array([
            theta,
            gamma,
            theta_dot,
            gamma_dot,
            self.plt_torque[0],
            self.plt_torque[1],
        ])

    @property
    def plt_torque(self):
        return self._plt_torque

    @plt_torque.setter
    def plt_torque(self, torques):
        self._plt_torque = torques


class HumanBalanceBulletEnv(MJCFBaseBulletEnv):
    def __init__(self, bsp=None, pltqs=None):
        self._timesteps = 0
        self._order = 0
        self._pltqs = pltqs
        self._pltq = None
        self.robot = HumanIDP()
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.81, timestep=0.01, frame_skip=1)

    def reset(self):
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)
        self._timesteps = 0
        self._order += 1
        self._set_pltqs()
        r = MJCFBaseBulletEnv.reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        return r

    def set_state(self, q, dq):
        self.robot.j1.set_state(q[0], dq[0])
        self.robot.j2.set_state(q[1], dq[1])
        self._set_pltqs()

    def step(self, a):
        prev_state = self.robot.calc_state()
        reward = -(prev_state[0] ** 2 + prev_state[1] ** 2
                   + 0.1 * prev_state[2] ** 2 + 0.1 * prev_state[3] ** 2)
        self._set_plt_torque()
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()
        self._timesteps += 1
        done = False
        self.HUD(state, a, done)
        return state, reward, done, {}

    def camera_adjust(self):
        self._p.resetDebugVisualizerCamera(2.4, -2.8, -27, [0, 0, 0.5])

    @property
    def current_obs(self):
        return self.robot.calc_state()

    @property
    def dt(self):
        return self.scene.dt

    @property
    def num_disturbs(self):
        return len(self._pltqs)

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def order(self):
        if self._pltqs is None:
            return self._order
        return self._order % len(self._pltqs)

    @property
    def pltq(self):
        return self._pltq

    def set_pltq(self, ext_pltq):
        if len(ext_pltq) == self.spec.max_episode_steps:
            self._pltq = ext_pltq
        else:
            raise TypeError("Input pltq length is wrong")

    def _set_pltqs(self):
        self._timesteps = 0
        if self._pltqs is not None:
            self._order = random.randrange(0, len(self._pltqs))
            self._pltq = self._pltqs[self.order] / self.robot.gear
        else:
            self._pltq = None
        self._set_plt_torque()

    def _set_plt_torque(self):
        if self.pltq is not None and self._timesteps != len(self.pltq):
            self.robot.plt_torque = self.pltq[self._timesteps, :].reshape(-1)
        else:
            self.robot.plt_torque = np.array([0, 0])
