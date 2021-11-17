import os

import gym
import mujoco_py
from copy import deepcopy
import random
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from xml.etree.ElementTree import ElementTree, parse


class IDPHuman(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, bsp=None, pltqs=None, init_states=None):
        self._timesteps = 0
        self._order = -1
        self._pltqs = pltqs
        self._pltq = None
        self._init_states = init_states
        filepath = os.path.join(os.path.dirname(__file__), "assets", "HPC_custom.xml")
        if bsp is not None:
            self._set_body_config(filepath, bsp)
        mujoco_env.MujocoEnv.__init__(self, filepath, frame_skip=5)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )
        utils.EzPickle.__init__(self)
        self.init_qpos = np.array([0.0, 0.0])
        self.init_qvel = np.array([0.0, 0.0])

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        for k, v in self.__dict__.items():
            if k not in ['model', 'sim', 'data']:
                setattr(result, k, deepcopy(v, memo))
        result.sim = mujoco_py.MjSim(result.model)
        result.data = result.sim.data
        return result

    def step(self, action: np.ndarray):
        prev_ob = self._get_obs()
        r = 1 - (prev_ob[0] ** 2 + prev_ob[1] ** 2 +
                 0.1 * prev_ob[2] ** 2 + 0.1 * prev_ob[3] ** 2 +
                 5e-6 * ((self.model.actuator_gear[0, 0] * action[0]) ** 2 +
                         (self.model.actuator_gear[0, 0] * action[1]) ** 2))
        self.do_simulation(action + self.plt_torque, self.frame_skip)
        self._timesteps += 1
        ob = self._get_obs()
        done = False
        info = {"obs": prev_ob.reshape(1, -1), "acts": action.reshape(1, -1)}
        return ob, r, done, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos,  # link angles
            self.sim.data.qvel,  # link angular velocities
            self.plt_torque,  # torque from platform movement
            np.array([self.timesteps / 600]),
        ]).ravel()

    def reset_model(self):
        self._order += 1
        if self._init_states is not None:
            self.init_qpos = np.array(self._init_states[self.order][:2])
            self.init_qvel = np.array(self._init_states[self.order][2:])
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self._set_pltqs()

    @property
    def current_obs(self):
        return self._get_obs()

    @property
    def num_disturbs(self):
        return len(self._pltqs)

    @property
    def order(self):
        if self._pltqs is None:
            return self._order
        return self._order % len(self._pltqs)

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def plt_torque(self):
        if self.pltq is not None:
            if self._timesteps == len(self.pltq):
                return np.array([0, 0])
            return self.pltq[self._timesteps, :].reshape(-1)
        else:
            return np.array([0, 0])

    @property
    def pltq(self):
        return self._pltq

    def set_pltq(self, ext_pltq):
        assert len(ext_pltq) == self.spec.max_episode_steps, "Input pltq length is wrong"
        self._pltq = ext_pltq

    def _set_pltqs(self):
        self._timesteps = 0
        if self._pltqs is not None:
            self._order = random.randrange(0, len(self._pltqs))
            self._pltq = self._pltqs[self.order] / self.model.actuator_gear[0, 0]
        else:
            self._pltq = None

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

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.5  # 0.12250000000000005  # v.model.stat.center[2]


class IDPHumanExp(IDPHuman):
    def _set_pltqs(self):
        self._timesteps = 0
        if self._pltqs is not None:
            self._pltq = self._pltqs[self.order] / self.model.actuator_gear[0, 0]
        else:
            self._pltq = None
