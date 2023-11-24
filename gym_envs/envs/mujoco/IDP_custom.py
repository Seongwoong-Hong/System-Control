import os

import mujoco_py
import numpy as np
from gym import utils, spaces
from xml.etree.ElementTree import ElementTree, parse

from gym_envs.envs.mujoco import BasePendulum


class IDPCustomDet(BasePendulum, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.high = np.array([0.2, 0.2, 1.0, 1.0])
        self.low = np.array([-0.2, -0.2, -1.0, -1.0])
        self.obs_target = np.zeros(4)
        filepath = os.path.join(os.path.dirname(__file__), "assets", "IDP_custom.xml")
        super(IDPCustomDet, self).__init__(filepath, *args, **kwargs)
        self.observation_space = spaces.Box(low=self.low, high=self.high)

    def step(self, obs_query: np.ndarray):
        prev_ob = self._get_obs()
        r = 0
        if self.timesteps % self._action_frame_skip == 0:
            self.obs_target[:2] = obs_query

        torque = np.array([0, 0])
        for segi in range(2):
            torque[segi] = (self.PDgain[0] * (self.obs_target[segi] - prev_ob[segi])
                            + self.PDgain[1] * (self.obs_target[segi + 2] - prev_ob[segi + 2]))
            torque[segi] /= self.model.actuator_gear[0, 0]
        torque = np.clip(torque, self.torque_space.low, self.torque_space.high)

        r += self.reward_fn(prev_ob, torque)

        ddx = self.ptb_acc[self.timesteps]
        self.data.qfrc_applied[:] = 0
        for idx, bodyName in enumerate(["leg", "body"]):
            body_id = self.model.body_name2id(bodyName)
            force_vector = np.array([-self.model.body_mass[body_id]*ddx, 0, 0])
            point = self.data.subtree_com[body_id]
            mujoco_py.functions.mj_applyFT(self.model, self.data, force_vector, np.zeros(3), point, body_id, self.sim.data.qfrc_applied)

        self.do_simulation(torque, self.frame_skip)
        ob = self._get_obs()
        done = ((ob[:2] < self.low[:2]).any() or (ob[:2] > self.high[:2]).any())
        if done:
            r -= 50

        self.timesteps += 1
        info = {'obs': prev_ob.reshape(1, -1), "acts": self.data.qfrc_actuator.copy()}
        return ob, r, done, info

    def reset_model(self):
        self.obs_target = np.zeros(4)
        return super(IDPCustomDet, self).reset_model()

    def reward_fn(self, ob, torque):
        r = 0
        r += np.exp(-200 * np.sum((ob[:2] - self._humanData[self.timesteps, :2]) ** 2)) \
            + 0.2 * np.exp(-1 * np.sum((ob[2:] - self._humanData[self.timesteps, 2:]) ** 2)) \
            - 0.1 * np.sum(torque ** 2)
        r += 0.1
        if self.ankle_max is not None:
            r -= 1/((np.abs(torque[0]) - self.ankle_max/self.model.actuator_gear[0, 0])**2 + 1e-2)
        return r

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

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.torque_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=4*self.low[:2], high=4*self.high[:2], dtype=np.float32)
        return self.action_space


class IDPCustom(IDPCustomDet):
    def step(self, obs_query: np.ndarray):
        r = 0
        self.obs_target[:2] = obs_query
        for _ in range(self._action_frame_skip):
            prev_ob = self._get_obs()
            torque = np.array([0, 0])
            for segi in range(2):
                torque[segi] = self.PDgain[0] * (self.obs_target[segi] - prev_ob[segi]) \
                               + self.PDgain[1] * (self.obs_target[segi + 2] - prev_ob[segi + 2])
            torque = np.clip(torque, self.torque_space.low, self.torque_space.high)

            r += self.reward_fn(prev_ob, torque)

            ddx = self.ptb_acc[self.timesteps]
            self.data.qfrc_applied[:] = 0
            for idx, bodyName in enumerate(["leg", "body"]):
                body_id = self.model.body_name2id(bodyName)
                force_vector = np.array([-self.model.body_mass[body_id] * ddx, 0, 0])
                point = self.data.subtree_com[body_id]
                mujoco_py.functions.mj_applyFT(
                    self.model, self.data, force_vector, np.zeros(3), point, body_id, self.sim.data.qfrc_applied)

            self.do_simulation(torque, self.frame_skip)
            ob = self._get_obs()
            done = ((ob[:2] < self.low[:2]).any() or (ob[:2] > self.high[:2]).any())
            if done:
                r -= 50
            if self.timesteps == 0:
                done = False
            self.timesteps += 1
            info = {'obs': prev_ob.reshape(1, -1), "acts": self.data.qfrc_actuator.copy()}
        return ob, r, done, info

    def reset_ptb(self):
        self._ptb_idx = self.np_random.choice(range(len(self._humanStates)))
        self._humanData = self._humanStates[self._ptb_idx]
        while self._humanData is None:
            self._ptb_idx = self.np_random.choice(range(len(self._humanStates)))
            self._humanData = self._humanStates[self._ptb_idx]
        st_time_dix = self.np_random.choice(range(self._epi_len))
        self._ptb_acc = np.append(self._ptb_acc, self._ptb_acc, axis=0)[st_time_dix:st_time_dix+self._epi_len]
        self._humanData = np.append(self._humanData, self._humanData, axis=0)[st_time_dix:st_time_dix+self._epi_len]