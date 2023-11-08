import os
import mujoco_py
import numpy as np
from copy import deepcopy
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from xml.etree.ElementTree import ElementTree, parse


class IDPCustom(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, bsp=None):
        self._order = 0
        self.timesteps = 0
        self.high = np.array([0.2, 0.3, 1.0, 1.0])
        self.low = np.array([-0.2, -0.3, -1.0, -1.0])
        filepath = os.path.join(os.path.dirname(__file__), "assets", "IDP_custom.xml")
        if bsp is not None:
            self._set_body_config(filepath, bsp)
        mujoco_env.MujocoEnv.__init__(self, filepath, frame_skip=1)
        utils.EzPickle.__init__(self)
        # self.observation_space = spaces.Box(low=self.low, high=self.high)
        self.timesteps = 0

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
        r = - (0.7139 * prev_ob[0] ** 2 + 0.5872182 * prev_ob[1] ** 2
               + 1.0639979 * prev_ob[2] ** 2 + 0.9540204 * prev_ob[3] ** 2
               + 1/3600 * 0.0061537065 * action[0] ** 2 + 1/2500 * 0.0031358577 * action[1] ** 2)
        r += 1

        ddx = np.sin(2*np.pi*self.timesteps/360)
        for idx, bodyName in enumerate(["leg", "body"]):
            body_id = self.model.body_name2id(bodyName)
            force_vector = np.array([-self.model.body_mass[body_id]*ddx, 0, 0])
            point = self.data.subtree_com[body_id]
            mujoco_py.functions.mj_applyFT(self.model, self.data, force_vector, np.zeros(3), point, body_id, self.sim.data.qfrc_applied)

        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        done = ((ob < self.low).any() or (ob > self.high).any()) and self.timesteps > 0
        # qpos = np.clip(ob[:2], a_min=self.low[:2], a_max=self.high[:2])
        # qvel = np.clip(ob[2:4], a_min=self.low[2:], a_max=self.high[2:])
        # self.set_state(qpos, qvel)
        # ob = self._get_obs()
        self.timesteps += 1
        info = {'obs': prev_ob.reshape(1, -1), "acts": action.reshape(1, -1)}
        return ob, r, False, info

    @property
    def order(self):
        return self._order

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos,  # link angles
            self.sim.data.qvel,   # link angular velocities
        ]).ravel()

    @property
    def current_obs(self):
        return self._get_obs()

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

    def reset_model(self):
        high = np.array([0.025, 0.025, 0.15, 0.2])
        low = -np.array([0.025, 0.075, 0.03, 0.3])
        init_state = self.np_random.uniform(low=low, high=high)
        self.set_state(init_state[:2], init_state[2:])
        self.timesteps = 0
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.5  # 0.12250000000000005  # v.model.stat.center[2]


class IDPCustomExp(IDPCustom):
    def __init__(self, init_states=None, bsp=None):
        super().__init__(bsp=bsp)
        self._init_states = init_states

    @property
    def init_states(self):
        if self._init_states is None:
            states = []
            for _ in range(100):
                states.append(self.observation_space.sample())
            self._init_states = states
        self._init_states = np.array(self._init_states)
        return self._init_states

    def reset_model(self):
        idx = self.np_random.randint(len(self.init_states))
        q = self.init_states[idx, :]
        self.set_state(q[:2], q[2:])
        return self._get_obs()


class IDPCustomDet(IDPCustomExp):
    def reset_model(self):
        self._order += 1
        q = self.init_states[self._order % len(self.init_states)]
        self.set_state(q[:2], q[2:])
        return self._get_obs()
