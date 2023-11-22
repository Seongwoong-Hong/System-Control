import os

import mujoco_py
import numpy as np
from gym import utils, spaces
from xml.etree.ElementTree import ElementTree, parse

from gym_envs.envs.mujoco import BasePendulum


class IPCustomDet(BasePendulum, utils.EzPickle):
    def __init__(self, bsp=None, humanStates=None):
        self.high = np.array([0.25, 1.0])
        self.low = np.array([-0.25, -1.0])
        filepath = os.path.join(os.path.dirname(__file__), "assets", "IP_custom.xml")
        super(IPCustomDet, self).__init__(filepath, humanStates)
        if bsp is not None:
            self._set_body_config(filepath, bsp)
        self.observation_space = spaces.Box(low=self.low, high=self.high)
        utils.EzPickle.__init__(self, bsp, humanStates)

    def step(self, action: np.ndarray):
        prev_ob = self._get_obs()
        torque = action
        rew = np.exp(-2*np.sum((prev_ob[:1] - self._humanData[self.timesteps, :1]) ** 2))\
            + np.exp(-0.1*np.sum((prev_ob[1:] - self._humanData[self.timesteps, 1:]) ** 2))\
            - 0.01 * np.sum(torque ** 2)
        rew += 0.1
        ddx = self.ptb_acc[self.timesteps]
        self.data.qfrc_applied[:] = 0
        for idx, bodyName in enumerate(["pole"]):
            body_id = self.model.body_name2id(bodyName)
            force_vector = np.array([-self.model.body_mass[body_id]*ddx, 0, 0])
            point = self.data.subtree_com[body_id]
            mujoco_py.functions.mj_applyFT(self.model, self.data, force_vector, np.zeros(3), point, body_id, self.data.qfrc_applied)
        self.do_simulation(torque, self.frame_skip)
        done = False
        ob = self._get_obs()
        if (ob[0] > (self.high[0] - 0.05)) or (ob[0] < (self.low[0] - 0.05)):
            done = True
            rew -= 50

        info = {"acts": self.data.qfrc_actuator.copy(), 'ptT': self.data.qfrc_applied.copy()}
        self.timesteps += 1
        return ob, rew, done, info

    @staticmethod
    def _set_body_config(filepath, bsp):
        m_u, l_u, com_u, I_u = bsp[6, :]
        m_s, l_s, com_s, I_s = bsp[2, :]
        m_t, l_t, com_t, I_t = bsp[3, :]
        l_l = l_s + l_t
        l = l_l + l_u
        m_l = 2 * (m_s + m_t)
        m = m_l + m_u
        com_l = (m_s * com_s + m_t * (l_s + com_t)) / (m_s + m_t)
        I_l = 2 * (I_s + m_s * (com_l - com_s) ** 2 + I_t + m_t * (com_l - (l_s + com_t)) ** 2)
        com = (m_l * com_l + m_u * (l_l + com_u)) / (m_l + m_u)
        I = I_l + m_l * (com - com_l) ** 2 + I_u + m_u * (com - (l_l + com_u)) ** 2
        tree = parse(filepath)
        root = tree.getroot()
        body = root.find("worldbody").find("body")
        body.find('geom').attrib['fromto'] = f"0 0 0 0 0 {l:.4f}"
        body.find('inertial').attrib['diaginertia'] = f"{I:.6f} {I:.6f} 0.001"
        body.find('inertial').attrib['mass'] = f"{m:.4f}"
        body.find('inertial').attrib['pos'] = f"0 0 {com:.4f}"
        m_tree = ElementTree(root)
        m_tree.write(filepath + ".tmp")
        os.replace(filepath + ".tmp", filepath)


class IPCustom(IPCustomDet):
    def reset_ptb(self):
        super().reset_ptb()
        st_time_dix = self.np_random.choice(range(self._epi_len))
        self._ptb_acc = np.append(self._ptb_acc, self._ptb_acc, axis=0)[st_time_dix:st_time_dix+self._epi_len]
        self._humanData = np.append(self._humanData, self._humanData, axis=0)[st_time_dix:st_time_dix+self._epi_len]
