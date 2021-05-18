import os
import random
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class IDPHuman(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, n_steps=None, bsp=None, pltqs=None):
        self._timesteps = 0
        self._order = 0
        self.n_steps = n_steps
        self._pltqs = pltqs
        self._pltq = None
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "HPC_custom.xml"), 5)
        utils.EzPickle.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, ))
        self.init_qpos = np.array([0.0, 0.0])
        self.init_qvel = np.array([0.0, 0.0])
        self._set_pltqs()
        if bsp is not None:
            self.bsp = bsp

    def step(self, action: np.ndarray):
        ob = self._get_obs()
        r = - (ob[0] ** 2 + ob[1] ** 2 + 1e-6 * action @ np.eye(2, 2) @ action.T)
        self.do_simulation(action + self.plt_torque, self.frame_skip)
        self._timesteps += 1
        ob = self._get_obs()
        done = False
        info = {}
        if self.n_steps is None:
            pass
        elif self._timesteps + 1 == self.n_steps:
            done = True
            info = {"terminal observation": ob}
            self._timesteps = 0
            self._order += 1
        return ob, r, done, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos,  # link angles
            self.sim.data.qvel,  # link angular velocities
            self.plt_torque,     # torque from platform movement
            # np.array([self.order])
        ]).ravel()

    def reset_model(self):
        self._timesteps = 0
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self._set_pltqs()

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
            return self.pltq[self._timesteps, :].reshape(-1)
        else:
            return np.array([0, 0])

    @property
    def pltq(self):
        return self._pltq

    @pltq.setter
    def pltq(self, ext_pltq):
        if len(ext_pltq) == self.n_steps:
            self._pltq = ext_pltq
        else:
            raise TypeError("Input pltq length is wrong")

    def _set_pltqs(self):
        self._timesteps = 0
        if self._pltqs is not None:
            self._order = random.randrange(0, len(self._pltqs))
            self._pltq = self._pltqs[self.order] / self.model.actuator_gear[0, 0]
        else:
            self._pltq = None

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.5  # 0.12250000000000005  # v.model.stat.center[2]


class IDPHumanExp(IDPHuman):
    def __init__(self, n_steps=None, bsp=None, pltqs=None):
        super().__init__(n_steps=n_steps, bsp=bsp)
        self._pltqs = pltqs
        self._set_pltqs()

    def _set_pltqs(self):
        self._timesteps = 0
        if self._pltqs is not None:
            self._pltq = self._pltqs[self.order] / self.model.actuator_gear[0, 0]
        else:
            self._pltq = None

    # def exp_isend(self):
    #     if self._order == len(self._pltqs):
    #         return True
    #     return False
