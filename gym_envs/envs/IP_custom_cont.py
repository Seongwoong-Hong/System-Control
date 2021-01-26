import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class IPCustomCont(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, n_steps=None):
        self.traj_len = 0
        self.n_steps = n_steps
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "IP_custom.xml"), 25)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        rew = - ob[0]**2 - ob[1]**2 - 0.001*(a*self.model.actuator_gear[0, 0])**2
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

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = 2 * self.model.stat.extent
