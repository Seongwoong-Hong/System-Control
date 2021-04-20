import os
import random
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class IDPHuman(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, n_steps=None, bsp=None, pltqs=None):
        self.traj_len = 0
        self.n_steps = n_steps
        self.pltqs = pltqs
        self.pltq = None
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "HPC_custom.xml"), 5)
        utils.EzPickle.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, ))
        self.init_qpos = np.array([0.0, 0.0])
        self.init_qvel = np.array([0.0, 0.0])
        self._set_pltqs()
        if bsp is not None:
            self.bsp = bsp
        self.traj_len = 0

    def step(self, action):
        self.do_simulation(action + self.plt_torque, self.frame_skip)
        ob = self._get_obs()
        r = - (ob[0] ** 2 + ob[1] ** 2 + 1e-6 * action @ np.eye(2, 2) @ action.T)
        done = False
        info = {}
        if self.n_steps is None:
            pass
        elif (self.traj_len + 1) == self.n_steps:
            done = True
            info = {"terminal observation": ob}
            self.traj_len = 0
        self.traj_len += 1
        return ob, r, done, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos,  # link angles
            self.sim.data.qvel,  # link angular velocities
            self.plt_torque,     # torque from platform movement
        ]).ravel()

    def reset_model(self):
        self.traj_len = 0
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self._set_pltqs()

    @property
    def plt_torque(self):
        if self.pltq is not None:
            return self.pltq[self.traj_len, :].reshape(-1)
        else:
            return np.array([0, 0])

    def _set_pltqs(self):
        if self.pltqs is not None:
            self.pltq = random.sample(self.pltqs, 1)[0] / self.model.actuator_gear[0, 0]
        else:
            self.pltq = None

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.5  # 0.12250000000000005  # v.model.stat.center[2]
