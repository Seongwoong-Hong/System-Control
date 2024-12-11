import os
import random
import string
import time
from pathlib import Path
from typing import List

import mujoco_py
import numpy as np
from gym import utils, spaces
from xml.etree.ElementTree import ElementTree, parse

from gym_envs.envs.mujoco import BasePendulum


class IDPMimicHumanDet(BasePendulum, utils.EzPickle):
    def __init__(self, limLevel=1, *args, **kwargs):
        self.high = np.deg2rad(np.array([45, 60, 60, 120]))
        self.low = np.deg2rad(np.array([-45, -120, -60.0, -120.0]))
        self.prev_torque = np.zeros(2, dtype=np.float32)
        self.limLevel = 10 ** (limLevel * ((-5) - (-2)) + (-2))
        filepath = os.path.join(os.path.dirname(__file__), "..", "assets", "IDP_custom.xml")
        super(IDPMimicHumanDet, self).__init__(filepath, *args, **kwargs)
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        utils.EzPickle.__init__(self, *args, **kwargs)

    def step(self, action):
        if self.delay and hasattr(self, "delayed_act"):
            assert action.shape == self.delayed_act[0].shape
            self.delayed_act = np.append(self.delayed_act, action[None, :], axis=0)
            action = self.delayed_act[0].copy()
            self.delayed_act = np.delete(self.delayed_act, 0, axis=0)
        return self.step_sims(action)

    def step_sims(self, action):
        prev_ob = self._get_obs()

        ddx = self.ptb_acc[self.timesteps]
        self.data.qfrc_applied[:] = 0.
        for idx, bodyName in enumerate(["leg", "body"]):
            body_id = self.model.body_name2id(bodyName)
            force_vector = np.array([-self.model.body_mass[body_id]*ddx, 0., 0.])
            # force_vector = np.array([0, 0., 0.])
            point = self.data.xipos[body_id]
            mujoco_py.functions.mj_applyFT(
                self.model, self.data, force_vector, np.zeros(3), point, body_id, self.sim.data.qfrc_applied)

        _action = -(np.array(self._jnt_stiffness)*prev_ob[:2] + np.array(self._jnt_damping)*prev_ob[2:]) / self.model.actuator_gear[:, 0]
        # torque = active_torque + passive_torque
        action = np.clip(_action + action, a_min=self.action_space.low, a_max=self.action_space.high)
        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        self.thacc = (ob[2:] - prev_ob[2:]) / self.dt
        passive_torque = -(np.array(self._jnt_stiffness)*prev_ob[:2] + np.array(self._jnt_damping)*prev_ob[2:])
        # torque = self.data.qfrc_actuator.copy() + passive_torque
        torque = self.data.qfrc_actuator[:].copy()
        done, r_penalty = self.is_done(ob, torque)
        r = self.reward_fn(prev_ob, torque / self.ankle_torque_max)
        r += r_penalty
        self.timesteps += 1
        if self.timesteps < 10:
            done = False
        self.prev_torque = self.data.qfrc_actuator.copy()
        comx, _, comy = self.data.subtree_com[0, :]
        info = {
            'prev_ob': prev_ob.reshape(1, -1),
            "torque": torque,
            "passive_torque": passive_torque,
            'comx': comx,
            'comy': comy,
            'ptb_acc': ddx,
        }
        return ob, r, done, info

    def reset_model(self):
        self.prev_torque = np.zeros(2, dtype=np.float32)
        ob = super(IDPMimicHumanDet, self).reset_model()
        if self.delay:
            _, lc1, lc2 = self.model.body_ipos[:, -1]
            _, m1, m2 = self.model.body_mass
            l1 = self.model.body_pos[2, 2]
            T2 = -m2*9.81*lc2*np.sin(ob[0] + ob[1])
            T1 = T2 - m1*9.81*lc1*np.sin(ob[0]) - m2*9.81*l1*np.sin(ob[0])
            pT = -(np.array(self._jnt_stiffness) * ob[:2] + np.array(self._jnt_damping) * ob[2:])
            self.delayed_act[:] = (np.array([T1, T2]) - pT) / self.model.actuator_gear[:, 0]
        return ob

    def reward_fn(self, ob, action):
        r = 0.
        r += np.exp(-20 * np.sum((ob[:2] - self._humanData[self.timesteps, :2]) ** 2)) \
            + np.exp(-2 * np.sum((ob[2:] - self._humanData[self.timesteps, 2:]) ** 2))
        r += 0.1

        return r

    def is_done(self, ob, torque):
        if self.timesteps < 10:
            return False, 0
        if np.any(ob[:2] <= self.low[:2]) or np.any(ob[:2] >= self.high[:2]):
            return True, -1
        if (np.abs(self.prev_torque - torque) >= 20).any() and self.torque_rate_limit:
            return True, -1
        return False, 0

    def _set_body_config(self, filepath, bsp):
        m_u, l_u, com_u, I_u = bsp[6, :]
        m_s, l_s, com_s, I_s = bsp[2, :]
        m_t, l_t, com_t, I_t = bsp[3, :]
        l_l = l_s + l_t
        m_l = 2 * (m_s + m_t)
        com_l = (m_s * com_s + m_t * (l_s + com_t)) / (m_s + m_t)
        I_l = 2 * (I_s + m_s * (com_l - com_s) ** 2 + I_t + m_t * (com_l - (l_s + com_t)) ** 2)
        tree = parse(filepath)
        root = tree.getroot()
        if self.use_seg_ang:
            root.find("compiler").attrib["coordinate"] = "global"
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
        if self._jnt_stiffness is not None:
            if not isinstance(self._jnt_stiffness, List):
                self._jnt_stiffness = [self._jnt_stiffness, self._jnt_stiffness]
        if self._jnt_damping is not None:
            if not isinstance(self._jnt_damping, List):
                self._jnt_damping = [self._jnt_damping, self._jnt_damping]

        if self.ankle_torque_max is not None:
            for motor in root.find('actuator').findall("motor"):
                if motor.get('name') == 'ank':
                    motor.attrib['gear'] = str(self.ankle_torque_max)
        m_tree = ElementTree(root)

        tmpname = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
        dirname = Path(filepath).parent / "tmp"
        dirname.mkdir(parents=False, exist_ok=True)
        while (dirname / (tmpname + ".xml")).is_file():
            tmpname = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
        filepath = str(dirname / (tmpname + ".xml"))

        m_tree.write(filepath)
        return filepath


class IDPMimicHuman(IDPMimicHumanDet):
    def reset_ptb(self):
        self._ptb_idx = self.np_random.choice(range(len(self._humanStates)))
        self._humanData = self._humanStates[self._ptb_idx]
        while self._humanData is None:
            self._ptb_idx = self.np_random.choice(range(len(self._humanStates)))
            self._humanData = self._humanStates[self._ptb_idx]
        self._set_ptb_acc()
        st_time_idx = self.np_random.choice(range(self._epi_len))
        self._ptb_acc = np.append(self._ptb_acc, self._ptb_acc, axis=0)[st_time_idx:st_time_idx+self._epi_len]
        self._humanData = np.append(self._humanData, self._humanData[-1, :] + self._humanData - self._humanData[0, :], axis=0)[st_time_idx:st_time_idx+self._epi_len]


class IDPMinEffortDet(IDPMimicHumanDet, utils.EzPickle):
    def __init__(self, tq_ratio: float = 0.5, ank_ratio: float = 0.5, vel_ratio: float = 0.5,
                 tqcost_ratio: float = 0.5, const_ratio: float = 1.0, ptb_range: List = None, *args, **kwargs):
        self.tq_ratio = tq_ratio
        self.tqcost_ratio = tqcost_ratio
        self.const_ratio = const_ratio
        self.ank_ratio = ank_ratio
        self.vel_ratio = vel_ratio
        assert 0 <= self.ank_ratio <= 1 and 0 <= self.tq_ratio <= 1 and 0 <= self.vel_ratio <= 1
        super(IDPMinEffortDet, self).__init__(*args, **kwargs)
        self._ptb_range = self._ptb_data_range.copy()
        if ptb_range is not None:
            self._ptb_range = np.array(ptb_range)
        utils.EzPickle.__init__(self, tq_ratio=self.tq_ratio, tqcost_ratio=self.tqcost_ratio, const_ratio=self.const_ratio,
                                ank_ratio=self.ank_ratio, vel_ratio=self.vel_ratio, ptb_range=self._ptb_range, *args, **kwargs)

    def reward_fn(self, ob, action):
        rew = 0.
        ahr = np.array([self.ank_ratio, 1 - self.ank_ratio])
        tahr = np.array([0.5, 1 - self.tq_ratio])
        rew -= ((1 - self.vel_ratio) * (ahr * (ob[:2] ** 2)).sum() + self.vel_ratio * (ob[2:] ** 2).sum())
        # rew -= ((1 - self.vel_ratio) * (ahr * (ob[:2] ** 2)).sum() + self.vel_ratio * (ob[2:] ** 2).sum())
        # rew -= ((1 - self.vel_ratio) * (self.data.subtree_com[0, 0] ** 2).sum() )
        rew -= self.tqcost_ratio * (tahr * (action ** 2)).sum()
        rew += 1
        return rew

    def is_done(self, ob, torque):
        done, r_penalty = IDPMimicHumanDet.is_done(self, ob, torque)
        if not done:
            # if (self._epi_len - self.timesteps)*self.dt < 0.5:
            #     if np.any(np.abs(ob[:2]) > 0.0873) or np.any(np.abs(ob[2:]) > 0.1):
            #         return True, -1
            if self.ankle_limit == "satu":
                return False, 0
            elif self.ankle_limit in ["hard", "soft"]:
                comx = self.data.subtree_com[0, 0]
                act = torque[0] / self.ankle_torque_max
                # if -0.075 < comx < 0.125:
                pmax, nmax = self.action_space.high[0], self.action_space.low[0]
                if act < pmax:
                # if (-0.5 < act < 1.0) and (-0.075 < comx < 0.15):
                    if self.ankle_limit == "soft":
                        postq = np.clip(act, a_min=0., a_max=pmax)
                        negtq = np.clip(act, a_min=nmax, a_max=0.)
                        r_penalty = 2*self.limLevel - self.limLevel * (
                                1 / ((postq / pmax - 1) ** 2 + self.limLevel) + 1 / ((negtq / nmax - 1) ** 2 + self.limLevel))
                        r_penalty *= self.const_ratio
                        # comx = min(comx, 0.12)
                        # r_penalty -= self.limLevel / ((comx / 0.12 - 1) ** 2 + self.limLevel)
                        # self.tq_ratio = 0.5 + 0.5 * self.const_ratio * self.limLevel * (
                        #         1 / ((postq / pmax - 1) ** 2 + self.limLevel) + 1 / ((negtq / nmax + 1) ** 2 + self.limLevel))
                    else:
                        r_penalty = 0
                    return False, r_penalty
                else:
                    return True, -1
            else:
                raise Exception(f"Unexpected ankle limit type: {self.ankle_limit}")
        return done, r_penalty


class IDPMinEffort(IDPMimicHuman, utils.EzPickle):
    def __init__(self, tq_ratio: float = 0.5, ank_ratio: float = 0.5, vel_ratio: float = 0.5,
                 tqcost_ratio: float = 0.5, const_ratio: float = 1.0, ptb_range: List = None, *args, **kwargs):
        self.tq_ratio = tq_ratio
        self.tqcost_ratio = tqcost_ratio
        self.const_ratio = const_ratio
        self.ank_ratio = ank_ratio
        self.vel_ratio = vel_ratio
        assert 0 <= self.ank_ratio <= 1 and 0 <= self.tq_ratio <= 1 and 0 <= self.vel_ratio <= 1
        super(IDPMinEffort, self).__init__(*args, **kwargs)
        self._ptb_range = self._ptb_data_range.copy()
        if ptb_range is not None:
            self._ptb_range = np.array(ptb_range)
        utils.EzPickle.__init__(self, tq_ratio=self.tq_ratio, tqcost_ratio=self.tqcost_ratio, const_ratio=self.const_ratio,
                                ank_ratio=self.ank_ratio, vel_ratio=self.vel_ratio, ptb_range=self._ptb_range, *args, **kwargs)

    def reward_fn(self, ob, action):
        return IDPMinEffortDet.reward_fn(self, ob, action)

    def reset_ptb(self):
        self._ptb_idx = self.np_random.choice(range(len(self._ptb_range)))
        x_max = -self._ptb_range[self._ptb_idx]  # Backward direction(-)
        fddx = self._cal_ptb_acc(x_max)
        self._humanData = np.zeros([self._epi_len, 4])
        # st_time_idx = self.np_random.choice(range(self._epi_len - round(2.5 / self.dt)))
        # st_time_idx = self.np_random.choice(range(round(self._epi_len - self._ptb_act_time/self.dt)))
        st_time_idx = 0
        _ptb_acc = np.append(np.zeros(st_time_idx), fddx)
        _ptb_acc = np.append(_ptb_acc, np.zeros(self._epi_len))
        self._ptb_acc = _ptb_acc[:self._epi_len]

    def reset_model(self):
        self.timesteps = 0
        self.reset_ptb()
        self.set_state(np.random.rand(self.model.nq) * np.deg2rad([2.5, 5]), np.zeros(self.model.nq))
        return self._get_obs()

    def is_done(self, ob, torque):
        return IDPMinEffortDet.is_done(self, ob, torque)

class IDPForwardPushDet(IDPMinEffortDet, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self._epi_len = 600
        self._ptb_force = np.zeros(self._epi_len)
        super(IDPForwardPushDet, self).__init__(*args, **kwargs)
        self._ptb_range = np.array([20, 40, 60, 80])

    def step_sims(self, action):
        prev_ob = self._get_obs()

        self.data.qfrc_applied[:] = 0.
        ptb_force = self.ptb_force[self.timesteps]
        for idx, bodyName in enumerate(["leg"]):
            body_id = self.model.body_name2id(bodyName)
            force_vector = np.array([ptb_force, 0., 0.])
            point = self.data.xipos[body_id]
            mujoco_py.functions.mj_applyFT(
                self.model, self.data, force_vector, np.zeros(3), point, body_id, self.sim.data.qfrc_applied)

        _action = -(np.array(self._jnt_stiffness)*prev_ob[:2] + np.array(self._jnt_damping)*prev_ob[2:]) / self.model.actuator_gear[:, 0]

        action = np.clip(_action + action, a_min=self.action_space.low, a_max=self.action_space.high)

        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        self.thacc = (ob[2:] - prev_ob[2:]) / self.dt
        passive_torque = -(np.array(self._jnt_stiffness)*prev_ob[:2] + np.array(self._jnt_damping)*prev_ob[2:])
        torque = self.data.qfrc_actuator[:].copy()
        done, r_penalty = self.is_done(ob, torque)
        r = self.reward_fn(prev_ob, torque / self.ankle_torque_max)
        r += r_penalty
        self.timesteps += 1
        if self.timesteps < 10:
            done = False
        self.prev_torque = self.data.qfrc_actuator.copy()
        comx, _, comy = self.data.subtree_com[0, :]
        info = {
            'prev_ob': prev_ob.reshape(1, -1),
            "torque": torque,
            "passive_torque": passive_torque,
            'comx': comx,
            'comy': comy,
            'ptb_force': ptb_force,
        }
        return ob, r, done, info

    def reset_ptb(self):
        self._ptb_idx = self._next_ptb_idx % len(self._ptb_range)
        self._humanData = np.zeros([self._epi_len, 4])
        self._next_ptb_idx += 1
        force = self._ptb_range[self._ptb_idx] * np.sin(5*np.pi * np.arange(0, 0.2, self.dt))
        self._ptb_force[40:40+len(force)] = force.copy()

    @property
    def ptb_force(self):
        return self._ptb_force


class IDPSinPtb(IDPMinEffort, utils.EzPickle):
    def __init__(self, tq_ratio: float = 0.5, ank_ratio: float = 0.5, vel_ratio: float = 0.5,
                 tqcost_ratio: float = 0.5, const_ratio: float = 1.0, frq_range: List = None, *args, **kwargs):
        self.tq_ratio = tq_ratio
        self.tqcost_ratio = tqcost_ratio
        self.const_ratio = const_ratio
        self.ank_ratio = ank_ratio
        self.vel_ratio = vel_ratio
        self._ptb = np.array([0.05])
        # self._frq_range = np.array([0.15, 0.45, 0.75, 1.05, 1.35])
        self._frq_range = np.array([0.3])
        if frq_range is not None:
            self._frq_range = np.array(frq_range)
        assert 0 <= self.ank_ratio <= 1 and 0 <= self.tq_ratio <= 1 and 0 <= self.vel_ratio <= 1
        self.high = np.deg2rad(np.array([50, 60, 60, 120]))
        self.low = np.deg2rad(np.array([-50, -120, -60.0, -120.0]))
        super(IDPSinPtb, self).__init__(*args, **kwargs)
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        utils.EzPickle.__init__(self, tq_ratio=self.tq_ratio, tqcost_ratio=self.tqcost_ratio, const_ratio=self.const_ratio,
                                ank_ratio=self.ank_ratio, vel_ratio=self.vel_ratio, frq_range=self._frq_range, *args, **kwargs)

    def reset_ptb(self):
        self._frq_idx = self.np_random.choice(range(len(self._frq_range)))
        self._ptb_acc = self._cal_ptb_acc(self._frq_range[self._frq_idx])
        self._humanData = np.zeros([self._epi_len, 4])

    def _cal_ptb_acc(self, frq):
        t = np.arange(0, self._epi_len) * self.dt
        return self._ptb * -4 * np.pi**2 * frq ** 2 * np.sin(2 * np.pi * frq * t)
        # return np.zeros(t.shape)


class IDPSinPtbDet(IDPSinPtb, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self._frq_idx = 0
        super(IDPSinPtbDet, self).__init__(*args, **kwargs)

    def reset_ptb(self):
        self._frq_idx = (self._frq_idx + 1) % len(self._frq_range)
        self._ptb_acc = self._cal_ptb_acc(self._frq_range[self._frq_idx])
        self._humanData = np.zeros([self._epi_len, 4])


class IDPMinMetCost(IDPMinEffort):
    def reward_fn(self, ob, action):
        rew = 0.
        if self.use_seg_ang:
            ob = convert_jntang_to_segang(ob)
        ahr = np.array([self.ank_ratio, 1 - self.ank_ratio])
        tahr = np.array([0.5, 1 - self.tq_ratio])

        rew -= ((1 - self.vel_ratio) * (ahr * (ob[:2] ** 2)).sum() + self.vel_ratio * (ob[2:] ** 2).sum())
        power = ob[2:] * (action * self.model.actuator_gear[:, 0]) * self.dt
        rew -= self.tqcost_ratio * (1 / 0.25 * power.dot(power > 0) + 1 / -1.2 * power.dot(power <= 0)) ** 2
        rew += 1
        return rew


class IDPHeadTrackDet(IDPMinEffortDet):
    def __init__(self, freqs=None, amp=None, *args, **kwargs):
        self._freqs = freqs
        if self._freqs is None:
            self._freqs = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5]
        self.amp = amp
        if self.amp is None:
            self.amp = 0.1
        self._frq_idx = 0
        self.freq = self._freqs[self._frq_idx]
        self.high = np.deg2rad(np.array([50, 60, 60, 120]))
        self.low = np.deg2rad(np.array([-50, -120, -60.0, -120.0]))
        super(IDPHeadTrackDet, self).__init__(*args, **kwargs)
        self.observation_space = spaces.Box(low=np.append(self.low, [-self.amp, -self.amp]), high=np.append(self.high, [self.amp, self.amp]), dtype=np.float32)
        utils.EzPickle.__init__(self, *args, **kwargs)

    def step_sims(self, action):
        prev_ob = self._get_obs()
        passive_torque = -(np.array(self._jnt_stiffness)*prev_ob[:2] + np.array(self._jnt_damping)*prev_ob[2:4])
        active_torque = action * self.model.actuator_gear[:, 0]
        torque = active_torque + passive_torque
        action = np.clip(torque / self.model.actuator_gear[:, 0], a_min=self.action_space.low, a_max=self.action_space.high)
        self.do_simulation(action, self.frame_skip)
        r = self.reward_fn(prev_ob, torque / self.ankle_torque_max)
        ob = self._get_obs()
        done, r_penalty = self.is_done(ob, torque)
        r += r_penalty
        self.timesteps += 1
        self.prev_torque = self.data.qfrc_actuator.copy()
        comx, _, comy = self.data.subtree_com[0, :]
        info = {
            'prev_ob': prev_ob.reshape(1, -1),
            "torque":  self.data.qfrc_actuator.copy(),
            "passive_torque": passive_torque,
            'comx': comx,
            'comy': comy,
            'headx': -(self.model.geom_pos[1, -1]*np.sin(ob[0]) + self.model.geom_pos[2, -1]*np.sin(ob[0]+ob[1])),
        }
        return ob, r, done, info

    def reset_model(self):
        self.freq = self._freqs[self._frq_idx]
        self._frq_idx = (self._frq_idx + 1) % len(self._freqs)
        super(IDPHeadTrackDet, self).reset_model()
        self.set_state(np.zeros(2), np.zeros(2))
        return self._get_obs()

    def reward_fn(self, ob, action):
        rew = 0.
        headx = - (self.model.geom_pos[1, -1]*np.sin(ob[0]) + self.model.geom_pos[2, -1]*np.sin(ob[0]+ob[1]))
        # rew += 1 / (1 + np.abs(headx - self.target_pos))
        rew += 1 - ((headx - self.target_pos) / self.amp) ** 2
        tahr = np.array([self.tq_ratio, 1 - self.tq_ratio])
        rew -= self.tqcost_ratio * (tahr * (action ** 2)).sum()
        rew += 1
        return rew

    def _get_obs(self):
        target_vel = self.amp*2*np.pi*self.freq*np.sin(2*np.pi*self.freq*self.timesteps*self.dt)
        ob = super()._get_obs()
        return np.append(ob, [self.target_pos, target_vel]).astype(np.float32).ravel()

    @property
    def target_pos(self):
        return self.amp - self.amp*np.cos(2*np.pi*self.freq*self.timesteps*self.dt)


class IDPHeadTrack(IDPHeadTrackDet):
    def reset_model(self):
        self._frq_idx = self.np_random.choice(range(len(self._freqs)))
        self.freq = self._freqs[self._frq_idx]
        return super(IDPHeadTrackDet, self).reset_model()


class IDPInitState(IDPMinEffort, utils.EzPickle):
    def __init__(self, tq_ratio: float = 0.5, ank_ratio: float = 0.5,
                 tqcost_ratio: float = 1.0, vel_ratio: float = 0.5, *args, **kwargs):
        self.tq_ratio = tq_ratio
        self.tqcost_ratio = tqcost_ratio
        self.ank_ratio = ank_ratio
        self.vel_ratio = vel_ratio
        assert 0 <= self.ank_ratio <= 1 and 0 <= self.tq_ratio <= 1 and 0 <= self.vel_ratio <= 1
        super(IDPInitState, self).__init__(*args, **kwargs)
        self._ptb_range = np.array([0])
        utils.EzPickle.__init__(self, tq_ratio=self.tq_ratio, tqcost_ratio=self.tqcost_ratio,
                                ank_ratio=self.ank_ratio, vel_ratio=self.vel_ratio, *args, **kwargs)

    def reset_model(self):
        self.timesteps = 0
        self.reset_ptb()
        init_state = self.np_random.uniform(self.low, self.high)
        self.set_state(init_state[:self.model.nq], init_state[self.model.nq:])
        return self._get_obs()


# ============================
# PD 제어기 환경
# ============================


class IDPPDMimicHumanDet(IDPMimicHumanDet, utils.EzPickle):
    def __init__(self, PDgain: List, *args, **kwargs):
        self._action_frame_skip = 4
        assert len(PDgain) == 2, "P, D gain은 각각 1개의 값이 입력되어야 합니다."
        self.PDgain = np.array(PDgain)
        self.obs_target = np.zeros(4, dtype=np.float32)
        super(IDPPDMimicHumanDet, self).__init__(*args, **kwargs)
        utils.EzPickle.__init__(self, *args, **kwargs)

    def step(self, obs_query: np.ndarray):
        if self.timesteps % self._action_frame_skip == 0:
            self.obs_target[:2] = obs_query
        return self.step_once()

    def step_once(self):
        prev_ob = self._get_obs()
        actuator_torque = np.array([0.0, 0.0])
        for segi in range(2):
            actuator_torque[segi] = (self.PDgain[0] * (self.obs_target[segi] - prev_ob[segi])
                                     + self.PDgain[1] * (self.obs_target[segi + 2] - prev_ob[segi + 2]))
        if self.delay:
            assert actuator_torque.shape == self.delayed_act[0].shape
            self.delayed_act = np.append(self.delayed_act, actuator_torque, axis=0)
            actuator_torque = self.delayed_act[0].copy()
            self.delayed_act = np.delete(self.delayed_act, 0, axis=0)
        action = np.clip(actuator_torque/self.model.actuator_gear[0, 0], self.torque_space.low, self.torque_space.high)
        return self.step_sims(action)

    def reset_model(self):
        self.obs_target = np.zeros(4, dtype=np.float32)
        return super(IDPMimicHumanDet, self).reset_model()

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.torque_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32)
        return self.action_space


class IDPPDMimicHuman(IDPPDMimicHumanDet):
    def step(self, obs_query: np.ndarray):
        rew = 0.
        dones = False
        self.obs_target[:2] = obs_query
        for _ in range(self._action_frame_skip):
            ob, r, done, info = self.step_once()
            dones = dones or done
            rew += r
        return ob, rew, dones, info

    def reset_ptb(self):
        IDPMimicHuman.reset_ptb(self)


class IDPPDMinEffortDet(IDPPDMimicHumanDet, utils.EzPickle):
    def __init__(self, cost_ratio=None, state_ratio=None, *args, **kwargs):
        self.cost_ratio = 0.5
        self.state_ratio = 0.5
        if cost_ratio is not None:
            self.cost_ratio = cost_ratio
        if state_ratio is not None:
            self.state_ratio = state_ratio
        super(IDPPDMinEffortDet, self).__init__(*args, **kwargs)
        utils.EzPickle.__init__(self, cost_ratio=self.cost_ratio, state_ratio=self.state_ratio, *args, **kwargs)

    def reward_fn(self, ob, action):
        rew = 0
        if self.use_seg_ang:
            ob = convert_jntang_to_segang(ob)
        rew -= self.cost_ratio * (np.array([self.state_ratio, 1-self.state_ratio]) @ (ob[:2] ** 2))
        rew -= (1 - self.cost_ratio) * (np.array([self.state_ratio, 1-self.state_ratio]) @ ((ob[2:]*action) ** 2))
        rew += 1
        if self.ankle_torque_max is not None:
            rew -= 1e-5 / ((np.abs(action)[0] - self.ankle_torque_max/self.model.actuator_gear[0, 0])**2 + 1e-5)
        return rew

    def is_done(self, ob, torque):
        return IDPMinEffort.is_done(self, ob, torque)


class IDPPDMinEffort(IDPPDMimicHuman, utils.EzPickle):
    def __init__(self, cost_ratio=None, state_ratio=None, *args, **kwargs):
        self.cost_ratio = 0.5
        self.state_ratio = 0.5
        if cost_ratio is not None:
            self.cost_ratio = cost_ratio
        if state_ratio is not None:
            self.state_ratio = state_ratio
        super(IDPPDMinEffort, self).__init__(*args, **kwargs)
        utils.EzPickle.__init__(self, cost_ratio=self.cost_ratio, state_ratio=self.state_ratio, *args, **kwargs)

    def reward_fn(self, ob, action):
        return IDPPDMinEffortDet.reward_fn(self, ob, action)

    def reset_ptb(self):
        IDPMinEffort.reset_ptb(self)

    def is_done(self, ob, torque):
        return IDPPDMinEffortDet.is_done(self, ob, torque)


class IDPPDMinMetCost(IDPPDMinEffort):
    def reward_fn(self, ob, action):
        rew = 0
        if self.use_seg_ang:
            ob = convert_jntang_to_segang(ob)
        rew -= self.cost_ratio * (np.array([self.state_ratio, 1-self.state_ratio]) @ (ob[:2] ** 2))
        power = np.array([self.state_ratio, 1-self.state_ratio])*ob[2:] * action
        rew -= (1-self.cost_ratio) * (1/0.25*power.dot(power > 0) + 1/-1.2*power.dot(power <= 0))
        rew += 1
        if self.ankle_torque_max is not None:
            rew -= 1e-5 / ((np.abs(action)[0] - self.ankle_torque_max/self.model.actuator_gear[0, 0])**2 + 1e-5)
        return rew


# ===================
# Local 함수
# ===================

def convert_jntang_to_segang(jntang):
    segang = jntang.copy()
    segang[1] += segang[0]
    segang[3] += segang[2]
    return segang


def normalize_variable(var, high, low):
    norm_var = (2*var - (high + low)) / (high - low)
    return norm_var
