import os
import numpy as np

from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.robot_bases import MJCFBasedRobot
from pybullet_envs.scene_abstract import SingleRobotEmptyScene


class InvertedDoublePendulum(MJCFBasedRobot):
    def __init__(self):
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mujoco", "assets", "IDP_custom.xml"))
        self.time = 0.0
        self.gear = 300
        MJCFBasedRobot.__init__(self, xml_path, 'IDP', action_dim=2, obs_dim=5)

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["hinge2"]
        self.time = 0.0
        u = self.np_random.uniform(low=-.1, high=.1, size=[2])
        du = self.np_random.uniform(low=-.1, high=.1, size=[2])
        self.j1.reset_current_position(float(u[0]), float(du[0]))
        self.j2.reset_current_position(float(u[1]), float(du[1]))
        self.j1.set_motor_torque(0)
        self.j2.set_motor_torque(0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.j1.set_motor_torque(self.gear * float(np.clip(a[0], -1, +1)))
        self.j2.set_motor_torque(self.gear * float(np.clip(a[1], -1, +1)))

    def calc_state(self):
        theta, theta_dot = self.j1.current_position()
        gamma, gamma_dot = self.j2.current_position()
        return np.array([
            theta,
            gamma,
            theta_dot,
            gamma_dot,
            self.time,
        ])


class InvertedDoublePendulumExp(InvertedDoublePendulum):
    def __init__(self):
        self._order = 0
        super().__init__()
        self.init_group = np.array([[[+0.10, +0.10], [+0.05, -0.05]],
                                    [[+0.15, +0.10], [-0.05, +0.05]],
                                    [[-0.16, +0.20], [+0.10, -0.10]],
                                    [[-0.10, +0.06], [+0.05, -0.10]],
                                    [[+0.05, +0.15], [-0.20, -0.20]],
                                    [[-0.05, +0.05], [+0.15, +0.15]],
                                    [[+0.12, +0.05], [-0.10, -0.15]],
                                    [[-0.08, +0.15], [+0.05, -0.15]],
                                    [[-0.15, +0.20], [-0.10, +0.05]],
                                    [[+0.20, +0.01], [+0.09, -0.15]],
                                    ])

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["hinge2"]
        u, du = self.init_group[self._order % len(self.init_group)]
        self.j1.reset_current_position(float(u[0]), float(du[0]))
        self.j2.reset_current_position(float(u[1]), float(du[1]))
        self.j1.set_motor_torque(0)
        self.j2.set_motor_torque(0)
        self._order += 1


class InvertedDoublePendulumBulletEnv(MJCFBaseBulletEnv):
    def __init__(self):
        self.robot = InvertedDoublePendulum()
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.81, timestep=0.001, frame_skip=8)

    def reset(self):
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)
        r = MJCFBaseBulletEnv.reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        return r

    def set_state(self, q, dq):
        self.robot.time = 0.0
        self.robot.j1.set_state(q[0], dq[0])
        self.robot.j2.set_state(q[1], dq[1])

    def step(self, a):
        prev_state = self.robot.calc_state()
        reward = 1 - (prev_state[0] ** 2 + prev_state[1] ** 2
                     + 0.1 * (prev_state[2] ** 2 + prev_state[3] ** 2)
                     + 1e-6 * ((self.robot.gear * a[0]) ** 2 + (self.robot.gear * a[1]) ** 2))
        self.robot.apply_action(a)
        self.scene.global_step()
        self.robot.time += self.dt / 4.8
        state = self.robot.calc_state()
        done = bool(
            0.95 <= state[0] or state[0] <= -0.95 or
            0.95 <= state[1] or state[1] <= -0.95
        )
        info = {'obs': prev_state.reshape(1, -1), "acts": a.reshape(1, -1)}
        if done:
            reward -= 1000
            info['done'] = done
        self.HUD(state, a, done)
        return state, reward, done, info

    def camera_adjust(self):
        self._p.resetDebugVisualizerCamera(2.4, -2.8, -27, [0, 0, 0.5])

    @property
    def current_obs(self):
        return self.robot.calc_state()

    @property
    def dt(self):
        return self.scene.dt


class InvertedDoublePendulumExpBulletEnv(InvertedDoublePendulumBulletEnv):
    def __init__(self):
        self.robot = InvertedDoublePendulumExp()
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def step(self, a):
        ns, r, done, info = super(InvertedDoublePendulumExpBulletEnv, self).step(a)
        return ns, r, None, info