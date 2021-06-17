import os
import numpy as np

from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.robot_bases import MJCFBasedRobot
from pybullet_envs.scene_abstract import SingleRobotEmptyScene


class InvertedDoublePendulum(MJCFBasedRobot):
    def __init__(self):
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mujoco", "assets", "IDP_custom.xml"))
        MJCFBasedRobot.__init__(self, xml_path, 'IDP', action_dim=2, obs_dim=4)

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.pole2 = self.parts["pole2"]
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["hinge2"]
        u = self.np_random.uniform(low=-.1, high=.1, size=[2])
        self.j1.reset_current_position(float(u[0]), 0)
        self.j2.reset_current_position(float(u[1]), 0)
        self.j1.set_motor_torque(0)
        self.j2.set_motor_torque(0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.j1.set_motor_torque(200 * float(np.clip(a[0], -1, +1)))
        self.j2.set_motor_torque(200 * float(np.clip(a[1], -1, +1)))

    def calc_state(self):
        theta, theta_dot = self.j1.current_position()
        gamma, gamma_dot = self.j2.current_position()
        return np.array([
            theta,
            gamma,
            theta_dot,
            gamma_dot,
        ])


class InvertedDoublePendulumBulletEnv(MJCFBaseBulletEnv):
    def __init__(self):
        self.robot = InvertedDoublePendulum()
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.81, timestep=0.01, frame_skip=1)

    def reset(self):
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)
        r = MJCFBaseBulletEnv.reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        return r

    def set_state(self, q, dq):
        self.robot.j1.set_state(q[0], dq[0])
        self.robot.j2.set_state(q[1], dq[1])

    def step(self, a):
        prev_state = self.robot.calc_state()
        reward = -(prev_state[0] ** 2 + prev_state[1] ** 2
                   + 0.1 * prev_state[2] ** 2 + 0.1 * prev_state[3] ** 2)
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()
        done = False
        self.HUD(state, a, done)
        return state, reward, done, {}

    def camera_adjust(self):
        self._p.resetDebugVisualizerCamera(2.4, -2.8, -27, [0, 0, 0.5])

    @property
    def current_obs(self):
        return self.robot.calc_state()
