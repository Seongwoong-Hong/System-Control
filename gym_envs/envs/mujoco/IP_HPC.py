import os
import mujoco_py
from copy import deepcopy
import random
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from xml.etree.ElementTree import ElementTree, parse


class IPHuman(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, bsp=None, pltqs=None, init_states=None):
        self._timesteps = 0
        self._order = -1
        self._pltqs = pltqs
        self._pltq = None
        self._init_states = init_states
        self.high = np.array([0.5, 2.0])
        self.low = np.array([-0.5, -2.0])
        filepath = os.path.join(os.path.dirname(__file__), "assets", "IP_HPC.xml")
        if bsp is not None:
            self._set_body_config(filepath, bsp)
        mujoco_env.MujocoEnv.__init__(self, filepath, frame_skip=1)
        utils.EzPickle.__init__(self)
        self.init_qpos = np.array([0.0])
        self.init_qvel = np.array([0.0])

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
        result.seed()
        return result

    def step(self, action: np.ndarray):
        prev_ob = self._get_obs()
        r = 1 - (3.5139 * prev_ob[0] ** 2 + 1.2872182 * prev_ob[1] ** 2 + 0.02537065 * action ** 2)
        self.do_simulation(action + self.plt_torque, self.frame_skip)
        self._timesteps += 1
        ob = self._get_obs()
        done = ((ob < self.low).any() or (ob > self.high).any()) and self.timesteps > 0
        # qpos = np.clip(ob[:2], a_min=self.low[:2], a_max=self.high[:2])
        # qvel = np.clip(ob[2:4], a_min=self.low[2:], a_max=self.high[2:])
        # self.set_state(qpos, qvel)
        # ob = self._get_obs()
        done = False
        info = {"obs": prev_ob.reshape(1, -1), "acts": action.reshape(1, -1)}
        return ob, r, done, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos,  # link angles
            self.sim.data.qvel,  # link angular velocities
        ]).ravel()

    def reset_model(self):
        self._order = random.randrange(0, len(self._pltqs))
        init_state = self.np_random.uniform(low=self.low, high=self.high)
        self.init_qpos, self.init_qvel = init_state[:1], init_state[1:]
        if self._init_states is not None:
            self.init_qpos = np.array(self._init_states[self.order][:1])
            self.init_qvel = np.array(self._init_states[self.order][1:2])
        self.set_state(self.init_qpos, self.init_qvel)
        self._set_pltqs()
        return self._get_obs()

    def set_state(self, *args):
        if len(args) == 1:
            qpos, qvel = args[0][:1], args[0][1:2]
        elif len(args) == 2:
            qpos, qvel = args[0], args[1]
        else:
            raise AssertionError
        super().set_state(qpos, qvel)

    @property
    def current_obs(self):
        return self._get_obs()

    @property
    def num_disturbs(self):
        return len(self._pltqs)

    @property
    def order(self):
        if self._pltqs is None:
            return self._order % len(self._init_states)
        return self._order % len(self._pltqs)

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def plt_torque(self):
        if self.pltq is not None:
            if self._timesteps == len(self.pltq):
                return np.array([0])
            return self.pltq[self._timesteps, :].reshape(-1)
        else:
            return np.array([0])

    @property
    def pltq(self):
        return self._pltq

    def set_pltq(self, ext_pltq):
        assert len(ext_pltq) == self.spec.max_episode_steps, "Input pltq length is wrong"
        self._pltq = ext_pltq

    def _set_pltqs(self):
        self._timesteps = 0
        if self._pltqs is not None:
            self._pltq = self._pltqs[self.order]
        else:
            self._pltq = None

    @staticmethod
    def _set_body_config(filepath, bsp):
        m, l, lc, I = bsp[0, :]
        tree = parse(filepath)
        root = tree.getroot()
        body = root.find("worldbody").find("body")
        body.find('geom').attrib['fromto'] = f"0 0 0 0 0 {l:.4f}"
        body.find('inertial').attrib['diaginertia'] = f"{I:.6f} {I:.6f} 0.001"
        body.find('inertial').attrib['mass'] = f"{m:.4f}"
        body.find('inertial').attrib['pos'] = f"0 0 {lc:.4f}"
        m_tree = ElementTree(root)
        m_tree.write(filepath + ".tmp")
        os.replace(filepath + ".tmp", filepath)

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.5  # 0.12250000000000005  # v.model.stat.center[2]


class IPHumanDet(IPHuman):
    def reset_model(self):
        self._order += 1
        init_state = self.np_random.uniform(low=self.low, high=self.high)
        self.init_qpos, self.init_qvel = init_state[:1], init_state[1:]
        if self._init_states is not None:
            self.init_qpos = np.array(self._init_states[self.order][:1])
            self.init_qvel = np.array(self._init_states[self.order][1:2])
        self.set_state(self.init_qpos, self.init_qvel)
        self._set_pltqs()
        return self._get_obs()
