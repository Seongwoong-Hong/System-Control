import os
from abc import ABCMeta

import mujoco_py
import numpy as np
from copy import deepcopy
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from xml.etree.ElementTree import ElementTree, parse

from gym_envs.envs.mujoco import BasePendulum


class IDPCustomDet(BasePendulum):
    def __init__(self, humanStates, bsp=None):
        self.high = np.array([0.2, 0.2])
        self.low = np.array([-0.2, -0.2])
        filepath = os.path.join(os.path.dirname(__file__), "assets", "IDP_custom.xml")
        if bsp is not None:
            self._set_body_config(filepath, bsp)
        super(IDPCustomDet, self).__init__(filepath, humanStates)
        self.observation_space = spaces.Box(low=self.low, high=self.high)

    def step(self, action: np.ndarray):
        prev_ob = self._get_obs()
        # r = -0.01 * np.sum(action ** 2)
        r = np.exp(-2 * np.sum((prev_ob[:2] - self._humanData[self.timesteps, :2]) ** 2)) \
            + np.exp(-0.1 * np.sum((prev_ob[2:] - self._humanData[self.timesteps, 2:]) ** 2)) \
            - 0.01 * np.sum(action ** 2)
        r += 0.1

        ddx = self.ptb_acc[self.timesteps]
        self.data.qfrc_applied[:] = 0
        for idx, bodyName in enumerate(["leg", "body"]):
            body_id = self.model.body_name2id(bodyName)
            force_vector = np.array([-self.model.body_mass[body_id]*ddx, 0, 0])
            point = self.data.subtree_com[body_id]
            mujoco_py.functions.mj_applyFT(self.model, self.data, force_vector, np.zeros(3), point, body_id, self.sim.data.qfrc_applied)

        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        done = ((ob[:2] < self.low).any() or (ob[:2] > self.high).any()) or (np.abs(action[0]) > 0.95)
        done = done and not self.timesteps == 0
        if done:
            r -= 50
        # qpos = np.clip(ob[:2], a_min=self.low[:2], a_max=self.high[:2])
        # qvel = np.clip(ob[2:4], a_min=self.low[2:], a_max=self.high[2:])
        # self.set_state(qpos, qvel)
        # ob = self._get_obs()
        self.timesteps += 1
        info = {'obs': prev_ob.reshape(1, -1), "acts": self.data.qfrc_actuator.copy()}
        return ob, r, done, info

    def set_state(self, *args):
        if len(args) == 1:
            qpos, qvel = args[0][:2], args[0][2:]
        elif len(args) == 2:
            qpos, qvel = args[0], args[1]
        else:
            raise AssertionError
        super().set_state(qpos, qvel)

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


class IDPCustom(IDPCustomDet):
    def reset_ptb(self):
        super().reset_ptb()
        st_time_dix = self.np_random.choice(range(self._epi_len))
        self._ptb_acc = np.append(self._ptb_acc, self._ptb_acc, axis=0)[st_time_dix:st_time_dix+self._epi_len]
        self._humanData = np.append(self._humanData, self._humanData, axis=0)[st_time_dix:st_time_dix+self._epi_len]