import os
import random
import numpy as np

from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.robot_bases import MJCFBasedRobot
from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from xml.etree.ElementTree import ElementTree, parse


class HumanIDP(MJCFBasedRobot):
    def __init__(self, bsp=None):
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mujoco", "assets", "HPC_custom.xml"))
        self.timesteps = 0
        self.gear = 300
        self.init_qpos = [0.0, 0.0]
        self.init_qvel = [0.0, 0.0]
        if bsp is not None:
            self._set_body_config(xml_path, bsp)
        MJCFBasedRobot.__init__(self, xml_path, 'HPC', action_dim=2, obs_dim=7)
        self._plt_torque = None

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["hinge2"]
        self.timesteps = 0
        self.j1.reset_current_position(self.init_qpos[0], self.init_qvel[0])
        self.j2.reset_current_position(self.init_qpos[1], self.init_qvel[1])
        self.j1.set_motor_torque(self.plt_torque[0])
        self.j2.set_motor_torque(self.plt_torque[1])
        return self.calc_state()

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.j1.set_motor_torque(self.gear * float(np.clip(a[0], -0.4, +0.4) + self.plt_torque[0]))
        self.j2.set_motor_torque(self.gear * float(np.clip(a[1], -0.4, +0.4) + self.plt_torque[1]))

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
            self.timesteps / 600,
        ])

    @staticmethod
    def _set_body_config(filepath, bsp):
        m_u, l_u, com_u, I_u = bsp[6, :]
        m_s, l_s, com_s, I_s = bsp[2, :]
        m_t, l_t, com_t, I_t = bsp[3, :]
        l_l = l_s + l_t
        m_l = 2 * (m_s + m_t)
        com_l = (m_s * com_s + m_t * (l_s + com_t)) / (m_s + m_t)
        I_l = 2 * (I_s + m_s * (com_l - com_s) ** 2 + I_t + m_t * (com_l - (l_s + com_t)) ** 2)
        tree = parse(filepath)
        root = tree.getroot()
        l_body = root.find("worldbody").find("body")
        l_body.find('geom').attrib['fromto'] = f"0 0 0 0 0 {l_l:.4f}"
        l_body.find('inertial').attrib['diaginertia'] = f"{I_l:.6f} {I_l:.6f} 0.001"
        l_body.find('inertial').attrib['mass'] = f"{m_l:.4f}"
        l_body.find('inertial').attrib['pos'] = f"0 0 {com_l:.4f}"
        u_body = l_body.find("body")
        u_body.attrib["pos"] = f"0 0 {l_l:.4f}"
        u_body.find("geom").attrib["fromto"] = f"0 0 0 0 0 {l_u:.4f}"
        u_body.find("inertial").attrib['diaginertia'] = f"{I_u:.6f} {I_u:.6f} 0.001"
        u_body.find("inertial").attrib['mass'] = f"{m_u:.4f}"
        u_body.find("inertial").attrib['pos'] = f"0 0 {com_u:.4f}"
        m_tree = ElementTree(root)
        m_tree.write(filepath + ".tmp")
        os.replace(filepath + ".tmp", filepath)

    @property
    def plt_torque(self):
        return self._plt_torque

    @plt_torque.setter
    def plt_torque(self, torques):
        self._plt_torque = torques


class HumanBalanceBulletEnv(MJCFBaseBulletEnv):
    def __init__(self, bsp=None, pltqs=None, init_states=None):
        self._order = -1
        self._pltqs = pltqs
        self._pltq = None
        self._init_states = init_states
        self.robot = HumanIDP(bsp=bsp)
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.81, timestep=0.0016666666666667, frame_skip=5)

    def reset(self):
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)
        self._order += 1
        self._set_pltqs()
        if self._init_states is not None:
            self.robot.init_qpos = self._init_states[self.order][:2]
            self.robot.init_qvel = self._init_states[self.order][2:]
        r = MJCFBaseBulletEnv.reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        return r

    def set_state(self, q, dq):
        self.robot.timesteps = 0
        self.robot.j1.set_state(q[0], dq[0])
        self.robot.j2.set_state(q[1], dq[1])
        self._set_pltqs()

    def step(self, a):
        prev_state = self.robot.calc_state()
        reward = 1 - (prev_state[0] ** 2 + prev_state[1] ** 2
                      + 0.1 * prev_state[2] ** 2 + 0.1 * prev_state[3] ** 2
                      + 5e-6 * (((self.robot.gear * a[0]) ** 2) + (self.robot.gear * a[1]) ** 2))
        self._set_plt_torque()
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()
        self.robot.timesteps += 1
        done = bool(
            0.95 <= state[0] or state[0] <= -0.95 or
            0.95 <= state[1] or state[1] <= -0.95
        )
        info = {'obs': prev_state.reshape(1, -1), "acts": a.reshape(1, -1)}
        self.HUD(state, a, done)
        return state, reward, None, info

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
        return self.robot.timesteps

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
        self.robot.timesteps = 0
        if self._pltqs is not None:
            self._order = random.randrange(0, len(self._pltqs))
            self._pltq = self._pltqs[self.order] / self.robot.gear
        else:
            self._pltq = None
        self._set_plt_torque()

    def _set_plt_torque(self):
        if self.pltq is not None and self.robot.timesteps != len(self.pltq):
            self.robot.plt_torque = self.pltq[self.robot.timesteps, :].reshape(-1)
        else:
            self.robot.plt_torque = np.array([0, 0])


class HumanBalanceExpBulletEnv(HumanBalanceBulletEnv):
    def _set_pltqs(self):
        self.robot.timesteps = 0
        if self._pltqs is not None:
            self._pltq = self._pltqs[self.order] / self.robot.gear
        else:
            self._pltq = None
        self._set_plt_torque()
