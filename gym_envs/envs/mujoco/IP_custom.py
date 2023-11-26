import os
import mujoco_py
import numpy as np
from gym import utils, spaces
from xml.etree.ElementTree import ElementTree, parse

from gym_envs.envs.mujoco import BasePendulum


class IPCustomDet(BasePendulum, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.high = np.array([0.25, 1.0])
        self.low = np.array([-0.25, -1.0])
        self.obs_target = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_torque = np.array([0.0], dtype=np.float32)
        filepath = os.path.join(os.path.dirname(__file__), "assets", "IP_custom.xml")
        super(IPCustomDet, self).__init__(filepath, *args, **kwargs)
        self.observation_space = spaces.Box(low=self.low, high=self.high)
        utils.EzPickle.__init__(self, *args, **kwargs)

    def step(self, obs_query: np.ndarray):
        if self.timesteps % self._action_frame_skip == 0:
            self.obs_target[0] = obs_query
        return self.step_once()

    def step_once(self):
        rew = 0
        prev_ob = self._get_obs()
        torque = self.PDgain @ (self.obs_target - prev_ob).T

        action = np.clip(torque / self.model.actuator_gear[0, 0], self.torque_space.low, self.torque_space.high)

        rew += self.reward_fn(prev_ob, action)

        ddx = self.ptb_acc[self.timesteps]
        self.data.qfrc_applied[:] = 0
        for idx, bodyName in enumerate(["pole"]):
            body_id = self.model.body_name2id(bodyName)
            force_vector = np.array([-self.model.body_mass[body_id]*ddx, 0, 0])
            point = self.data.subtree_com[body_id]
            mujoco_py.functions.mj_applyFT(self.model, self.data, force_vector, np.zeros(3), point, body_id, self.data.qfrc_applied)

        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        done = False
        if (ob[0] > (self.high[0] - 0.05)) or (ob[0] < (self.low[0] - 0.05)):
            done = True
        if np.abs(self.prev_torque - torque) >= 7:
            done = True
        if self.timesteps < 10:
            done = False

        self.timesteps += 1
        info = {"torque": self.data.qfrc_actuator.copy(), 'ptT': self.data.qfrc_applied.copy()}
        self.prev_torque = torque

        return ob, rew, done, info

    def reset_model(self):
        self.obs_target = np.array([0.0, 0.0], dtype=np.float32)
        return super(IPCustomDet, self).reset_model()

    def reward_fn(self, ob, action):
        rew = 0
        rew += np.exp(-200 * (ob[0] - self._humanData[self.timesteps, 0]) ** 2) \
            + 0.2 * np.exp(-1 * (ob[1] - self._humanData[self.timesteps, 1]) ** 2)
        rew += 0.1
        if self.ankle_torque_max is not None:
            rew -= 1e-5 / ((np.abs(action)[0] - self.ankle_torque_max/self.model.actuator_gear[0, 0])**2 + 1e-5)
        return rew

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

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.torque_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        return self.action_space


class IPCustom(IPCustomDet):
    def step(self, obs_query: np.ndarray):
        rew = 0
        self.obs_target[0] = obs_query
        for _ in range(self._action_frame_skip):
            ob, r, done, info = self.step_once()
            rew += r
        return ob, rew, done, info

    def reset_ptb(self):
        self._ptb_idx = self.np_random.choice(range(len(self._humanStates)))
        self._humanData = self._humanStates[self._ptb_idx]
        while self._humanData is None:
            self._ptb_idx = self.np_random.choice(range(len(self._humanStates)))
            self._humanData = self._humanStates[self._ptb_idx]
        st_time_dix = self.np_random.choice(range(self._epi_len))
        self._ptb_acc = np.append(self._ptb_acc, self._ptb_acc, axis=0)[st_time_dix:st_time_dix+self._epi_len]
        self._humanData = np.append(self._humanData, self._humanData, axis=0)[st_time_dix:st_time_dix+self._epi_len]
