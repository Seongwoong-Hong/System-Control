import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class IPCustom(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, n_steps=None):
        self.traj_len = 0
        self.n_steps = n_steps
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "IP_custom.xml"), 20)

    def step(self, a):
        ob = self._get_obs()
        rew = - ob[0] ** 2
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        info = {}
        if self.n_steps is None:
            pass
        elif self.traj_len == self.n_steps:
            done = True
            info = {"terminal observation": ob}
            self.traj_len = 0
        self.traj_len += 1
        return ob, rew, done, info

    def reset_model(self):
        q = np.random.uniform(size=2, low=-0.25, high=0.25)
        self.set_state(np.array([q[0]]), np.array([q[1]]))
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    @property
    def current_obs(self):
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = 2 * self.model.stat.extent


class IPCustomExp(IPCustom):
    def __init__(self, n_steps=None):
        super().__init__(n_steps=n_steps)
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