import os
import mujoco_py
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class IPCustom(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.timesteps = 0
        self.high = np.array([0.2, 0.3])
        self.low = np.array([-0.2, -0.3])
        self._ptb_acc = np.zeros(360)
        self._ptb_range = np.array([3, 4.5, 6, 7.5, 9])
        self._maxT = 3
        self._ptbT = 0.3
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "IP_custom.xml"), 1)
        self.observation_space = spaces.Box(low=self.low, high=self.high)

    def step(self, action: np.ndarray):
        prev_ob = self._get_obs()
        rew = - 0.7139 * prev_ob[0] ** 2 - 1.0639979 * prev_ob[1] ** 2 - 1600 * 0.0061537065 * action[0] ** 2

        ddx = self.ptb_force[self.timesteps]
        for idx, bodyName in enumerate(["pole"]):
            body_id = self.model.body_name2id(bodyName)
            force_vector = np.array([-self.model.body_mass[body_id]*ddx, 0, 0])
            point = self.data.subtree_com[body_id]
            mujoco_py.functions.mj_applyFT(self.model, self.data, force_vector, np.zeros(3), point, body_id, self.data.qfrc_applied)
        # qpos = np.clip(ob[:1], a_min=np.array([-0.1]), a_max=np.array([0.1]))
        # qvel = np.clip(ob[1:], a_min=np.array([-0.3]), a_max=np.array([0.3]))
        # self.set_state(qpos, qvel)
        ob = self._get_obs()
        info = {'obs': prev_ob.reshape(1, -1), "acts": action.reshape(1, -1)}
        self.timesteps += 1
        return ob, rew, False, info

    def reset_model(self):
        init_state = self.np_random.uniform(low=self.low, high=self.high)
        self.set_state(init_state[:1], init_state[1:])
        self.timesteps = 0
        self.reset_ptb()
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def reset_ptb(self):
        acc_max = self.np_random.choice(self._ptb_range)
        ptb_act_t = 0.275
        self._ptb_acc = np.zeros(int(self._ptbT / self.dt))
        self._ptb_acc = np.append(self._ptb_acc, 6*acc_max/ptb_act_t**2 * (1 - 2*np.linspace(0, 1, int(ptb_act_t/self.dt))))
        self._ptb_acc = np.append(self._ptb_acc, acc_max * np.ones(int((self._maxT - self._ptbT - ptb_act_t)/self.dt)))
        self._ptb_acc = -self._ptb_acc

    @property
    def current_obs(self):
        return self._get_obs()

    @property
    def ptb_acc(self):
        return self._ptb_acc[self.timesteps]

    @property
    def ptb_force(self):
        return self._ptb_acc

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = 2 * self.model.stat.extent


class IPCustomDet(IPCustom):
    def __init__(self):
        super().__init__()
        self.init_group = np.array([[+0.30, -0.15],
                                    [-0.30, +0.15],
                                    [+0.15, +0.15],
                                    [-0.15, -0.15],
                                    [+0.12, +0.26],
                                    [-0.12, -0.26],
                                    [+0.28, -0.15],
                                    [-0.28, +0.15],
                                    [-0.20, -0.25],
                                    [+0.20, +0.25]])
        self.i = 0

    def reset_model(self):
        if self.i >= len(self.init_group):
            self.i = 0
        q = self.init_group[self.i]
        self.set_state(np.array([q[0]]), np.array([q[1]]))
        self.i += 1
        return self._get_obs()
